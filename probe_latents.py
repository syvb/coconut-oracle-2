"""
Probe whether latent reasoning steps encode numbers/math ops relevant to
the computation.

For each latent step, we decode the hidden state two ways:
  1. Cosine similarity to the token embedding matrix (what the latent "looks like")
  2. Logits from the LM head (what the model would predict next)

We then check whether the top-decoded tokens are numbers that correspond to
intermediate values in the gold chain-of-thought solution.
"""

import json
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from datasets import load_dataset
from codi_model import load_model, NUM_LATENT, device
import re

console = Console()

# ── helpers ──────────────────────────────────────────────────────────────

def extract_intermediate_numbers(solution_text):
    """Pull every number that appears in the GSM8K chain-of-thought solution."""
    text = solution_text.replace(",", "")
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", text)]


def top_tokens_by_cosine(embedding, embed_weight, tokenizer, k=15):
    """Return top-k tokens by cosine similarity to the latent vector."""
    emb = embedding.squeeze().float()
    W = embed_weight.float()
    cos = F.cosine_similarity(emb.unsqueeze(0), W, dim=1)
    topk = torch.topk(cos, k)
    results = []
    for score, idx in zip(topk.values, topk.indices):
        tok = tokenizer.decode([idx.item()])
        results.append((tok, score.item()))
    return results


def top_tokens_by_logits(hidden_state, model, tokenizer, ori_vocab, k=15):
    """Run hidden state through lm_head to get logit-based top-k tokens."""
    h = hidden_state.squeeze().unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
    logits = model.lm_head(h)[:, -1, :ori_vocab]
    topk = torch.topk(logits.squeeze(), k)
    results = []
    for score, idx in zip(topk.values, topk.indices):
        tok = tokenizer.decode([idx.item()])
        results.append((tok, score.item()))
    return results


def token_is_number(tok_str):
    """Check if a token string decodes to a number."""
    tok_str = tok_str.strip()
    try:
        float(tok_str.replace(",", ""))
        return True
    except ValueError:
        return False


def number_from_token(tok_str):
    """Extract number from token string, or None."""
    tok_str = tok_str.strip()
    try:
        return float(tok_str.replace(",", ""))
    except ValueError:
        return None


def numbers_in_topk(top_tokens):
    """Return set of numbers found in top-k token list."""
    nums = set()
    for tok, _ in top_tokens:
        n = number_from_token(tok)
        if n is not None:
            nums.add(n)
    return nums


def number_overlap(decoded_nums, gold_nums):
    """Fraction of decoded numbers that appear in the gold solution numbers."""
    if not decoded_nums:
        return 0.0
    hits = sum(1 for n in decoded_nums if n in gold_nums)
    return hits / len(decoded_nums)


# ── main experiment ──────────────────────────────────────────────────────

@torch.no_grad()
def probe_question(model, prj, tokenizer, bot_id, eot_id, question, solution):
    """Run inference on one question, probing each latent step."""
    embed_fn = model.model.embed_tokens
    embed_weight = embed_fn.weight
    ori_vocab = model.config.vocab_size - 3

    gold_nums = set(extract_intermediate_numbers(solution))

    # Format prompt same as benchmark
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    bot_tensor = torch.tensor([[bot_id]], dtype=torch.long, device=device)
    input_ids = torch.cat([inputs["input_ids"], bot_tensor], dim=1)

    # Encode
    outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    past_kv = outputs.past_key_values
    hidden = outputs.hidden_states[-1][:, -1, :]   # pre-projection hidden
    latent_embd = prj(hidden.unsqueeze(1))

    steps = []

    for i in range(NUM_LATENT):
        outputs = model(
            inputs_embeds=latent_embd, use_cache=True,
            output_hidden_states=True, past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        hidden = outputs.hidden_states[-1][:, -1, :]
        latent_embd = prj(hidden.unsqueeze(1))

        cosine_top = top_tokens_by_cosine(latent_embd, embed_weight, tokenizer, k=15)
        logits_top = top_tokens_by_logits(hidden, model, tokenizer, ori_vocab, k=15)

        cos_nums = numbers_in_topk(cosine_top)
        log_nums = numbers_in_topk(logits_top)

        steps.append({
            "step": i + 1,
            "cosine_top": cosine_top,
            "logits_top": logits_top,
            "cosine_nums": cos_nums,
            "logits_nums": log_nums,
            "cos_overlap": number_overlap(cos_nums, gold_nums),
            "log_overlap": number_overlap(log_nums, gold_nums),
        })

    return steps, gold_nums


def display_question(idx, question, solution, gold_nums, steps):
    """Pretty-print one question's probe results."""
    console.rule(f"[bold] Question {idx}")
    console.print(Panel(question, title="Question", border_style="blue"))
    console.print(f"[dim]Gold intermediate numbers:[/] {sorted(gold_nums)}\n")

    for s in steps:
        table = Table(title=f"Latent Step {s['step']}", show_lines=True)
        table.add_column("Method", style="bold")
        table.add_column("Top tokens", max_width=70)
        table.add_column("Numbers found")
        table.add_column("Overlap w/ gold")

        cos_toks = "  ".join(
            f"[bold green]{t}[/]({sc:.2f})" if token_is_number(t)
            else f"{t}({sc:.2f})"
            for t, sc in s["cosine_top"][:10]
        )
        log_toks = "  ".join(
            f"[bold green]{t}[/]({sc:.1f})" if token_is_number(t)
            else f"{t}({sc:.1f})"
            for t, sc in s["logits_top"][:10]
        )

        table.add_row(
            "Cosine→Embed",
            cos_toks,
            str(sorted(s["cosine_nums"])),
            f"{s['cos_overlap']:.0%}",
        )
        table.add_row(
            "Logits→LMHead",
            log_toks,
            str(sorted(s["logits_nums"])),
            f"{s['log_overlap']:.0%}",
        )
        console.print(table)
    console.print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10, help="number of questions to probe")
    parser.add_argument("--offset", type=int, default=0, help="start index in test set")
    parser.add_argument("--json", type=str, default=None, help="save raw stats to JSON")
    args = parser.parse_args()

    model, prj, tokenizer, bot_id, eot_id = load_model()
    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Aggregate stats
    all_cos_overlaps = []
    all_log_overlaps = []
    all_cos_num_frac = []  # fraction of top-10 that are numbers
    all_log_num_frac = []
    records = []

    for i in range(args.offset, min(args.offset + args.n, len(ds))):
        q = ds[i]["question"]
        sol = ds[i]["answer"]
        steps, gold_nums = probe_question(model, prj, tokenizer, bot_id, eot_id, q, sol)
        display_question(i, q, sol, gold_nums, steps)

        for s in steps:
            all_cos_overlaps.append(s["cos_overlap"])
            all_log_overlaps.append(s["log_overlap"])
            cos_num_count = sum(1 for t, _ in s["cosine_top"][:10] if token_is_number(t))
            log_num_count = sum(1 for t, _ in s["logits_top"][:10] if token_is_number(t))
            all_cos_num_frac.append(cos_num_count / 10)
            all_log_num_frac.append(log_num_count / 10)

        records.append({
            "index": i,
            "question": q,
            "gold_numbers": sorted(gold_nums),
            "steps": [
                {
                    "step": s["step"],
                    "cosine_top10": [(t, round(sc, 3)) for t, sc in s["cosine_top"][:10]],
                    "logits_top10": [(t, round(sc, 1)) for t, sc in s["logits_top"][:10]],
                    "cosine_numbers": sorted(s["cosine_nums"]),
                    "logits_numbers": sorted(s["logits_nums"]),
                    "cos_overlap": round(s["cos_overlap"], 3),
                    "log_overlap": round(s["log_overlap"], 3),
                }
                for s in steps
            ],
        })

    # Summary
    console.rule("[bold] SUMMARY")
    summary = Table(title="Aggregate Statistics")
    summary.add_column("Metric")
    summary.add_column("Cosine→Embed")
    summary.add_column("Logits→LMHead")

    n_steps = len(all_cos_overlaps)
    summary.add_row(
        "Avg fraction of top-10 that are numbers",
        f"{sum(all_cos_num_frac)/n_steps:.1%}",
        f"{sum(all_log_num_frac)/n_steps:.1%}",
    )
    summary.add_row(
        "Avg overlap of decoded nums with gold CoT",
        f"{sum(all_cos_overlaps)/n_steps:.1%}",
        f"{sum(all_log_overlaps)/n_steps:.1%}",
    )

    # Per-step breakdown
    for step in range(1, NUM_LATENT + 1):
        cos_frac = [all_cos_num_frac[j] for j in range(n_steps) if j % NUM_LATENT == step - 1]
        log_frac = [all_log_num_frac[j] for j in range(n_steps) if j % NUM_LATENT == step - 1]
        cos_ol = [all_cos_overlaps[j] for j in range(n_steps) if j % NUM_LATENT == step - 1]
        log_ol = [all_log_overlaps[j] for j in range(n_steps) if j % NUM_LATENT == step - 1]
        summary.add_row(
            f"  Step {step}: num frac / gold overlap",
            f"{sum(cos_frac)/len(cos_frac):.0%} / {sum(cos_ol)/len(cos_ol):.0%}",
            f"{sum(log_frac)/len(log_frac):.0%} / {sum(log_ol)/len(log_ol):.0%}",
        )

    console.print(summary)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(records, f, indent=2)
        console.print(f"\nSaved detailed results to {args.json}")


if __name__ == "__main__":
    main()
