"""
Test whether latent steps selectively boost correct numbers/operators.

For each latent step we get the full logit distribution and ask:
  1. Are number tokens that appear in the gold CoT ranked higher / have
     higher logits than number tokens that do NOT appear?
  2. Are the math operators used in the gold CoT (+, -, *, /) boosted
     relative to the ones not used?

This goes beyond "are there numbers in top-k" to test actual selectivity.
"""

import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datasets import load_dataset
from codi_model import load_model, NUM_LATENT, device

console = Console()


# ── Build token index ────────────────────────────────────────────────────

def build_number_token_map(tokenizer, ori_vocab):
    """Map number values -> list of token ids that decode to that number.

    We consider tokens whose stripped decode is a pure integer 0-999.
    """
    num_to_ids = {}  # float -> [token_ids]
    id_to_num = {}   # token_id -> float
    for tid in range(ori_vocab):
        tok_str = tokenizer.decode([tid]).strip()
        # Match integers and simple decimals
        if re.fullmatch(r"-?\d+\.?\d*", tok_str):
            try:
                val = float(tok_str)
            except ValueError:
                continue
            num_to_ids.setdefault(val, []).append(tid)
            id_to_num[tid] = val
    return num_to_ids, id_to_num


def find_operator_tokens(tokenizer, ori_vocab):
    """Find token ids for +, -, *, / (exact single-char tokens)."""
    ops = {}
    for tid in range(ori_vocab):
        tok_str = tokenizer.decode([tid])
        # We want tokens that are just the operator, possibly with leading space
        stripped = tok_str.strip()
        if stripped in ("+", "-", "*", "/", "×", "÷"):
            canonical = {"+": "+", "-": "-", "*": "*", "/": "/",
                         "×": "*", "÷": "/"}
            op = canonical.get(stripped, stripped)
            ops.setdefault(op, []).append(tid)
    return ops  # {"+": [id1, id2, ...], "-": [...], ...}


def extract_gold_numbers(solution_text):
    """All numbers in the gold chain-of-thought."""
    text = solution_text.replace(",", "")
    return set(float(n) for n in re.findall(r"-?\d+\.?\d*", text))


def extract_gold_operators(solution_text):
    """Which of +, -, *, / appear in the gold solution."""
    ops = set()
    if "+" in solution_text:
        ops.add("+")
    if "-" in solution_text:
        ops.add("-")
    if "*" in solution_text or "×" in solution_text:
        ops.add("*")
    if "/" in solution_text or "÷" in solution_text:
        ops.add("/")
    return ops


def parse_computation_steps(solution_text):
    """Extract ordered computation steps from GSM8K solutions.

    Returns list of dicts with keys: operands (set of floats),
    result (float), operator (str).
    Parses patterns like "16 - 3 - 4 = 9" and "9 * 2 = 18".
    """
    steps = []
    # Match lines like "number op number (op number)* = result"
    for line in solution_text.split("\n"):
        line = line.strip()
        # Find equations with = sign
        m = re.search(r'([\d,.]+(?:\s*[+\-*/×÷]\s*[\d,.]+)+)\s*=\s*([\d,.]+)', line)
        if m:
            expr, result = m.group(1), m.group(2)
            result = float(result.replace(",", ""))
            # Extract all numbers in the expression
            operands = set(float(n.replace(",", "")) for n in re.findall(r'[\d,.]+', expr))
            # Extract operators
            ops_found = set(re.findall(r'[+\-*/×÷]', expr))
            canonical = {"+": "+", "-": "-", "*": "*", "/": "/", "×": "*", "÷": "/"}
            ops_canonical = set(canonical.get(o, o) for o in ops_found)
            steps.append({
                "operands": operands,
                "result": result,
                "operators": ops_canonical,
                "all_nums": operands | {result},
            })
    return steps


# ── Core probing ─────────────────────────────────────────────────────────

@torch.no_grad()
def probe_question(model, prj, tokenizer, bot_id, eot_id,
                   question, solution, id_to_num, op_token_map, ori_vocab):
    """Probe one question. Returns per-step stats."""
    embed_fn = model.model.embed_tokens

    gold_nums = extract_gold_numbers(solution)
    gold_ops = extract_gold_operators(solution)
    comp_steps = parse_computation_steps(solution)
    non_gold_ops = set(op_token_map.keys()) - gold_ops

    prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
              f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    bot_tensor = torch.tensor([[bot_id]], dtype=torch.long, device=device)
    input_ids = torch.cat([inputs["input_ids"], bot_tensor], dim=1)

    outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    past_kv = outputs.past_key_values
    latent_embd = prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))

    step_results = []

    for i in range(NUM_LATENT):
        outputs = model(
            inputs_embeds=latent_embd, use_cache=True,
            output_hidden_states=True, past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        hidden = outputs.hidden_states[-1][:, -1, :]
        latent_embd = prj(hidden.unsqueeze(1))

        # Full logits from lm_head
        logits = model.lm_head(hidden.unsqueeze(0).unsqueeze(0))[:, -1, :ori_vocab]
        logits = logits.squeeze().float().cpu()

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Ranks (0 = highest)
        ranks = torch.zeros_like(logits, dtype=torch.long)
        sorted_idx = torch.argsort(logits, descending=True)
        ranks[sorted_idx] = torch.arange(len(logits))

        # ── Number analysis ──
        # For each number token, get its logit, prob, rank
        gold_logits = []
        nongold_logits = []
        gold_ranks = []
        nongold_ranks = []
        gold_probs = []
        nongold_probs = []

        for tid, num_val in id_to_num.items():
            l = logits[tid].item()
            r = ranks[tid].item()
            p = probs[tid].item()
            if num_val in gold_nums:
                gold_logits.append(l)
                gold_ranks.append(r)
                gold_probs.append(p)
            else:
                nongold_logits.append(l)
                nongold_ranks.append(r)
                nongold_probs.append(p)

        # ── Operator analysis ──
        op_stats = {}
        for op_name, tids in op_token_map.items():
            # Max logit across token variants for this operator
            op_logit = max(logits[t].item() for t in tids)
            op_prob = max(probs[t].item() for t in tids)
            op_rank = min(ranks[t].item() for t in tids)  # best rank
            op_stats[op_name] = {
                "logit": op_logit, "prob": op_prob, "rank": op_rank,
                "is_gold": op_name in gold_ops,
            }

        gold_op_logits = [v["logit"] for v in op_stats.values() if v["is_gold"]]
        nongold_op_logits = [v["logit"] for v in op_stats.values() if not v["is_gold"]]
        gold_op_ranks = [v["rank"] for v in op_stats.values() if v["is_gold"]]
        nongold_op_ranks = [v["rank"] for v in op_stats.values() if not v["is_gold"]]

        # ── Per-computation-step analysis ──
        # For each parsed equation, check if the result number is boosted
        # at this latent step
        comp_step_ranks = []
        for cs in comp_steps:
            result_val = cs["result"]
            # Find best rank for result token
            result_tids = [tid for tid, v in id_to_num.items() if v == result_val]
            if result_tids:
                best_rank = min(ranks[t].item() for t in result_tids)
                best_logit = max(logits[t].item() for t in result_tids)
                comp_step_ranks.append({
                    "result": result_val,
                    "rank": best_rank,
                    "logit": best_logit,
                })

        step_results.append({
            "step": i + 1,
            # Numbers
            "gold_num_mean_logit": np.mean(gold_logits) if gold_logits else float("nan"),
            "nongold_num_mean_logit": np.mean(nongold_logits) if nongold_logits else float("nan"),
            "gold_num_median_rank": np.median(gold_ranks) if gold_ranks else float("nan"),
            "nongold_num_median_rank": np.median(nongold_ranks) if nongold_ranks else float("nan"),
            "gold_num_total_prob": sum(gold_probs),
            "nongold_num_total_prob": sum(nongold_probs),
            "n_gold_nums": len(gold_logits),
            "n_nongold_nums": len(nongold_logits),
            # Operators
            "op_stats": op_stats,
            "gold_op_mean_logit": np.mean(gold_op_logits) if gold_op_logits else float("nan"),
            "nongold_op_mean_logit": np.mean(nongold_op_logits) if nongold_op_logits else float("nan"),
            "gold_op_mean_rank": np.mean(gold_op_ranks) if gold_op_ranks else float("nan"),
            "nongold_op_mean_rank": np.mean(nongold_op_ranks) if nongold_op_ranks else float("nan"),
            # Per-equation result tracking
            "comp_step_ranks": comp_step_ranks,
        })

    return step_results, gold_nums, gold_ops, comp_steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=30, help="number of questions")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    model, prj, tokenizer, bot_id, eot_id = load_model()
    ori_vocab = model.config.vocab_size - 3

    console.print("Building number/operator token index...")
    num_to_ids, id_to_num = build_number_token_map(tokenizer, ori_vocab)
    op_token_map = find_operator_tokens(tokenizer, ori_vocab)
    console.print(f"  {len(id_to_num)} number tokens covering {len(num_to_ids)} distinct values")
    console.print(f"  Operators: { {k: len(v) for k, v in op_token_map.items()} }")

    ds = load_dataset("openai/gsm8k", "main", split="test")

    # Accumulators per step
    agg = {s: {
        "gold_logits": [], "nongold_logits": [],
        "gold_ranks": [], "nongold_ranks": [],
        "gold_probs": [], "nongold_probs": [],
        "gold_op_logits": [], "nongold_op_logits": [],
        "gold_op_ranks": [], "nongold_op_ranks": [],
        "result_ranks_by_eq_idx": {},  # eq_index -> [ranks across questions]
    } for s in range(1, NUM_LATENT + 1)}

    records = []

    for qi in range(args.offset, min(args.offset + args.n, len(ds))):
        q = ds[qi]["question"]
        sol = ds[qi]["answer"]
        steps, gold_nums, gold_ops, comp_steps = probe_question(
            model, prj, tokenizer, bot_id, eot_id,
            q, sol, id_to_num, op_token_map, ori_vocab,
        )

        if qi < 3:
            # Print detailed output for first few
            console.rule(f"[bold] Q{qi}")
            console.print(f"[dim]{q[:120]}...[/]")
            console.print(f"Gold nums: {sorted(gold_nums)}  |  Gold ops: {gold_ops}")
            if comp_steps:
                eq_strs = [f"{sorted(cs['operands'])} {cs['operators']} -> {cs['result']}" for cs in comp_steps]
            console.print(f"Equations: {eq_strs}")
            for s in steps:
                delta_l = s["gold_num_mean_logit"] - s["nongold_num_mean_logit"]
                console.print(
                    f"  Step {s['step']}: "
                    f"num logit gold={s['gold_num_mean_logit']:+.1f} vs non={s['nongold_num_mean_logit']:+.1f} "
                    f"(Δ={delta_l:+.1f})  |  "
                    f"rank gold={s['gold_num_median_rank']:.0f} vs non={s['nongold_num_median_rank']:.0f}  |  "
                    f"prob gold={s['gold_num_total_prob']:.3f} vs non={s['nongold_num_total_prob']:.3f}"
                )
                # Show per-equation result ranks
                for cr in s["comp_step_ranks"]:
                    console.print(f"    eq result {cr['result']}: rank={cr['rank']}, logit={cr['logit']:.1f}")
                # Operators
                for op, st in s["op_stats"].items():
                    tag = "[bold green]GOLD[/]" if st["is_gold"] else "[dim]non [/]"
                    console.print(f"    op '{op}' {tag}: logit={st['logit']:.1f} rank={st['rank']}")

        # Accumulate
        for s in steps:
            a = agg[s["step"]]
            a["gold_logits"].append(s["gold_num_mean_logit"])
            a["nongold_logits"].append(s["nongold_num_mean_logit"])
            a["gold_ranks"].append(s["gold_num_median_rank"])
            a["nongold_ranks"].append(s["nongold_num_median_rank"])
            a["gold_probs"].append(s["gold_num_total_prob"])
            a["nongold_probs"].append(s["nongold_num_total_prob"])
            if not np.isnan(s["gold_op_mean_logit"]):
                a["gold_op_logits"].append(s["gold_op_mean_logit"])
            if not np.isnan(s["nongold_op_mean_logit"]):
                a["nongold_op_logits"].append(s["nongold_op_mean_logit"])
            if not np.isnan(s["gold_op_mean_rank"]):
                a["gold_op_ranks"].append(s["gold_op_mean_rank"])
            if not np.isnan(s["nongold_op_mean_rank"]):
                a["nongold_op_ranks"].append(s["nongold_op_mean_rank"])
            for ei, cr in enumerate(s["comp_step_ranks"]):
                a["result_ranks_by_eq_idx"].setdefault(ei, []).append(cr["rank"])

        records.append({
            "index": qi, "question": q,
            "gold_numbers": sorted(gold_nums),
            "gold_operators": sorted(gold_ops),
            "steps": [
                {k: v for k, v in s.items() if k != "op_stats"}
                for s in steps
            ],
        })

        if (qi + 1) % 10 == 0:
            console.print(f"  ... processed {qi + 1 - args.offset}/{args.n}")

    # ── Summary tables ───────────────────────────────────────────────────
    console.rule("[bold] NUMBER SELECTIVITY")
    t = Table(title="Gold CoT numbers vs non-gold numbers (LM head logits)",
              show_lines=True)
    t.add_column("Step")
    t.add_column("Gold mean logit")
    t.add_column("Non-gold mean logit")
    t.add_column("Δ logit")
    t.add_column("Gold median rank")
    t.add_column("Non-gold median rank")
    t.add_column("Gold total prob")
    t.add_column("Non-gold total prob")

    for step in range(1, NUM_LATENT + 1):
        a = agg[step]
        gl = np.nanmean(a["gold_logits"])
        nl = np.nanmean(a["nongold_logits"])
        gr = np.nanmean(a["gold_ranks"])
        nr = np.nanmean(a["nongold_ranks"])
        gp = np.nanmean(a["gold_probs"])
        np_ = np.nanmean(a["nongold_probs"])
        delta = gl - nl
        style = "bold green" if delta > 0.5 else ("bold red" if delta < -0.5 else "dim")
        t.add_row(
            str(step),
            f"{gl:.2f}", f"{nl:.2f}",
            f"[{style}]{delta:+.2f}[/{style}]",
            f"{gr:.0f}", f"{nr:.0f}",
            f"{gp:.4f}", f"{np_:.4f}",
        )

    # Overall
    all_gl = np.nanmean([v for s in agg.values() for v in s["gold_logits"]])
    all_nl = np.nanmean([v for s in agg.values() for v in s["nongold_logits"]])
    all_gr = np.nanmean([v for s in agg.values() for v in s["gold_ranks"]])
    all_nr = np.nanmean([v for s in agg.values() for v in s["nongold_ranks"]])
    all_gp = np.nanmean([v for s in agg.values() for v in s["gold_probs"]])
    all_np = np.nanmean([v for s in agg.values() for v in s["nongold_probs"]])
    d = all_gl - all_nl
    t.add_row(
        "[bold]ALL[/]",
        f"{all_gl:.2f}", f"{all_nl:.2f}", f"[bold]{d:+.2f}[/]",
        f"{all_gr:.0f}", f"{all_nr:.0f}",
        f"{all_gp:.4f}", f"{all_np:.4f}",
    )
    console.print(t)

    # ── Operator table ───────────────────────────────────────────────────
    console.rule("[bold] OPERATOR SELECTIVITY")
    t2 = Table(title="Gold CoT operators vs non-gold operators (LM head logits)",
               show_lines=True)
    t2.add_column("Step")
    t2.add_column("Gold op mean logit")
    t2.add_column("Non-gold op mean logit")
    t2.add_column("Δ logit")
    t2.add_column("Gold op mean rank")
    t2.add_column("Non-gold op mean rank")

    for step in range(1, NUM_LATENT + 1):
        a = agg[step]
        gl = np.mean(a["gold_op_logits"]) if a["gold_op_logits"] else float("nan")
        nl = np.mean(a["nongold_op_logits"]) if a["nongold_op_logits"] else float("nan")
        gr = np.mean(a["gold_op_ranks"]) if a["gold_op_ranks"] else float("nan")
        nr = np.mean(a["nongold_op_ranks"]) if a["nongold_op_ranks"] else float("nan")
        delta = gl - nl if not (np.isnan(gl) or np.isnan(nl)) else float("nan")
        style = "bold green" if delta > 0.5 else ("bold red" if delta < -0.5 else "dim")
        t2.add_row(
            str(step),
            f"{gl:.2f}", f"{nl:.2f}",
            f"[{style}]{delta:+.2f}[/{style}]",
            f"{gr:.0f}", f"{nr:.0f}",
        )
    console.print(t2)

    # ── Per-equation-index rank evolution ─────────────────────────────────
    console.rule("[bold] EQUATION RESULT RANK BY LATENT STEP")
    console.print("For each equation index in the CoT (eq0=first, eq1=second, ...),")
    console.print("show the median rank of that equation's result number at each latent step.\n")

    t3 = Table(title="Median rank of equation result at each latent step")
    t3.add_column("Eq idx")
    for s in range(1, NUM_LATENT + 1):
        t3.add_column(f"Step {s}")

    max_eq = max(max(a["result_ranks_by_eq_idx"].keys(), default=-1) for a in agg.values())
    for ei in range(max_eq + 1):
        row = [f"eq{ei}"]
        for step in range(1, NUM_LATENT + 1):
            vals = agg[step]["result_ranks_by_eq_idx"].get(ei, [])
            if vals:
                row.append(f"{np.median(vals):.0f}")
            else:
                row.append("-")
        t3.add_row(*row)
    console.print(t3)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(records, f, indent=2, default=str)
        console.print(f"\nSaved to {args.json}")


if __name__ == "__main__":
    main()
