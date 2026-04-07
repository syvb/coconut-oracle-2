#!/usr/bin/env python3
"""
Interactive chat with CoDi (Continuous Chain-of-Thought) LLaMA 1B.

The model performs latent reasoning in continuous embedding space before
generating its answer. This script visualizes those hidden "thinking" steps.
"""

import argparse
import sys
import time
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.console import Group
from rich import box

from codi_model import load_model, generate_with_traces, NUM_LATENT, device

console = Console()


# ── Visualization ─────────────────────────────────────────────────────────────

def latent_heatmap(latent_embd, width=72):
    """Render the latent vector as a 2-row colored heatmap using half-block chars.

    Downsamples the 2048-dim vector to width*2 bins, arranges as 2 rows,
    and uses '▀' with fg=top_row color, bg=bottom_row color to draw
    two pixel rows per text line. Diverging blue-white-red colormap.
    """
    flat = latent_embd.squeeze().float()
    n = flat.shape[0]
    # Downsample to width*2 values via averaging
    bins = width * 2
    chunk = n // bins
    vals = flat[:chunk * bins].view(bins, chunk).mean(dim=1)

    # Normalize to [-1, 1] range using percentile-ish clamping
    vmax = vals.abs().quantile(0.95).item() + 1e-8
    vals = (vals / vmax).clamp(-1, 1).tolist()

    top_row = vals[:width]
    bot_row = vals[width:]

    def val_to_rgb(v):
        """Diverging colormap: blue (neg) → light gray (zero) → red (pos), light-mode friendly."""
        if v >= 0:
            t = v
            r = int(180 + 75 * t)       # 180 → 255
            g = int(180 - 140 * t)       # 180 → 40
            b = int(180 - 150 * t)       # 180 → 30
        else:
            t = -v
            r = int(180 - 160 * t)       # 180 → 20
            g = int(180 - 110 * t)       # 180 → 70
            b = int(180 + 55 * t)        # 180 → 235
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    parts = []
    for t, b_val in zip(top_row, bot_row):
        tr, tg, tb = val_to_rgb(t)
        br, bg, bb = val_to_rgb(b_val)
        parts.append(f"[rgb({tr},{tg},{tb}) on rgb({br},{bg},{bb})]▀[/]")
    return "".join(parts)


def diff_heatmap(curr, prev, width=72):
    """Render the change between two latent vectors as a heatmap."""
    diff = (curr.squeeze().float() - prev.squeeze().float())
    n = diff.shape[0]
    bins = width
    chunk = n // bins
    vals = diff[:chunk * bins].view(bins, chunk).mean(dim=1)
    vmax = vals.abs().quantile(0.95).item() + 1e-8
    vals = (vals / vmax).clamp(-1, 1).tolist()

    def val_to_rgb(v):
        if v >= 0:
            t = v
            r = int(220 - 30 * t)
            g = int(220 - 180 * t)
            b = int(220 - 190 * t)
        else:
            t = -v
            r = int(220 - 200 * t)
            g = int(220 - 150 * t)
            b = int(220 - 10 * t)
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))

    parts = []
    for v in vals:
        cr, cg, cb = val_to_rgb(v)
        parts.append(f"[on rgb({cr},{cg},{cb})] [/]")
    return "".join(parts)


def make_latent_panel(step, latent_embd, prev_embd, embed_weight, tokenizer, elapsed):
    norm = latent_embd.squeeze().float().norm().item()
    flat = latent_embd.squeeze().float()
    sparsity = (flat.abs() < 0.01).float().mean().item()

    # Drift from previous latent
    drift_line = ""
    if prev_embd is not None:
        sim = F.cosine_similarity(
            latent_embd.flatten().float(), prev_embd.flatten().float(), dim=0
        ).item()
        drift_line = f"  cos(prev)=[bold]{sim:.4f}[/]"

    stats_line = (
        f"  ‖h‖=[bold rgb(180,120,0)]{norm:.1f}[/]  "
        f"sparsity=[bold rgb(180,120,0)]{sparsity:.1%}[/]  "
        f"latency=[bold rgb(0,120,60)]{elapsed*1000:.0f}ms[/]"
    )
    if drift_line:
        stats_line += drift_line

    heatmap = latent_heatmap(latent_embd, width=72)
    body = f"{stats_line}\n  {heatmap}"

    if prev_embd is not None:
        dhm = diff_heatmap(latent_embd, prev_embd, width=72)
        body += f"\n  {dhm}  [rgb(100,100,100)]delta[/]"

    palette = [
        "rgb(150,30,150)",   # purple
        "rgb(0,80,180)",     # blue
        "rgb(0,130,130)",    # teal
        "rgb(0,130,50)",     # green
        "rgb(180,120,0)",    # amber
        "rgb(190,40,40)",    # red
    ]
    c = palette[step % len(palette)]
    glyphs = "◆◇●○★◈"
    g = glyphs[step % len(glyphs)]

    return Panel(
        body,
        title=f"[bold {c}]{g} Thought {step + 1}/{NUM_LATENT}[/]",
        border_style=c,
        box=box.HEAVY,
        width=80,
    )


def thinking_progress(step, total):
    parts = []
    for i in range(total):
        if i <= step:
            parts.append("[bold rgb(150,30,150)]◆[/]")
        else:
            parts.append("[rgb(200,180,210)]◇[/]")
    bar = "  ".join(parts)
    return Align.center(Text.from_markup(f"latent reasoning: {bar}"))


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_response(model, prj, tokenizer, bot_id, eot_id, prompt,
                      max_new_tokens=256, temperature=0.1, greedy=True,
                      no_reasoning=False):
    embed_fn = model.model.embed_tokens
    embed_weight = embed_fn.weight.data
    ori_vocab = embed_weight.shape[0] - 3

    # Format: <question><bot_id>  (remove_eos=True means no eos between question and bot)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    bot_tensor = torch.tensor([[bot_id]], dtype=torch.long, device=device)
    input_ids = torch.cat([inputs["input_ids"], bot_tensor], dim=1)

    # Phase 1: Encode question
    t0 = time.time()
    outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    past_kv = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    latent_embd = prj(latent_embd)
    encode_time = time.time() - t0

    if no_reasoning:
        # Skip latent thinking entirely
        console.print()
        console.print(Panel(
            f"Encoded [bold]{len(input_ids[0])}[/] tokens in "
            f"[bold rgb(0,120,60)]{encode_time * 1000:.0f}ms[/]. "
            f"[bold rgb(190,40,40)]Skipping latent reasoning (--no-reasoning)[/]",
            title="[bold rgb(190,40,40)]⟐ CoDi (no reasoning)[/]",
            border_style="rgb(190,40,40)", box=box.DOUBLE, width=80,
        ))
    else:
        console.print()
        console.print(Panel(
            f"Encoded [bold]{len(input_ids[0])}[/] tokens in "
            f"[bold rgb(0,120,60)]{encode_time * 1000:.0f}ms[/]. "
            f"Entering latent reasoning ({NUM_LATENT} steps)...",
            title="[bold rgb(0,80,180)]⟐ CoDi Latent Reasoning[/]",
            border_style="rgb(0,80,180)", box=box.DOUBLE, width=80,
        ))

        # Phase 2: Latent thinking loop
        panels = []
        prev_embd = None

        with Live(console=console, refresh_per_second=30) as live:
            for i in range(NUM_LATENT):
                t_step = time.time()
                outputs = model(
                    inputs_embeds=latent_embd, use_cache=True,
                    output_hidden_states=True, past_key_values=past_kv,
                )
                past_kv = outputs.past_key_values
                new_latent = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                new_latent = prj(new_latent)
                elapsed = time.time() - t_step

                panel = make_latent_panel(i, new_latent, prev_embd, embed_weight, tokenizer, elapsed)
                panels.append(panel)
                prev_embd = latent_embd.clone()
                latent_embd = new_latent

                prog = thinking_progress(i, NUM_LATENT)
                live.update(Group(*panels, prog))
                time.sleep(0.1)

    # Phase 3: Decode (with just <eot_id>, no eos — matches remove_eos=True)
    eot_emb = embed_fn(
        torch.tensor([[eot_id]], dtype=torch.long, device=device)
    )

    console.print()
    console.print("[bold rgb(0,130,50)]▸ Answer:[/] ", end="")

    output_emb = eot_emb
    generated_tokens = []

    for _ in range(max_new_tokens):
        out = model(inputs_embeds=output_emb, use_cache=True, past_key_values=past_kv)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :ori_vocab]

        if greedy or temperature <= 0:
            next_id = torch.argmax(logits, dim=-1)
        else:
            logits /= temperature
            top_k = 40
            top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
            logits[logits < top_k_vals[:, -1:]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        tok_id = next_id.item()
        if tok_id == tokenizer.eos_token_id:
            break
        # Also stop on other special tokens
        if tok_id >= ori_vocab:
            break

        token_str = tokenizer.decode([tok_id])
        console.print(token_str, end="", highlight=False)
        generated_tokens.append(tok_id)
        output_emb = embed_fn(next_id.unsqueeze(0))

    console.print()
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

BANNER_BODY = (
    "[bold rgb(150,30,150)]◆[/] [bold]CoDi[/] — Chain of Thought → Continuous Latent Space\n"
    "\n"
    "Model: [bold rgb(0,80,180)]bcywinski/codi_llama1b-answer_only[/]\n"
    "Base:  [bold rgb(0,80,180)]LLaMA 3.2 1B Instruct + LoRA (r=128)[/]\n"
    "Steps: [bold rgb(150,30,150)]6 continuous latent reasoning tokens[/]\n"
    "\n"
    "Trained on GSM8K math. Each [bold rgb(150,30,150)]◆[/] is a hidden thought\n"
    "vector projected to token space for visualization.\n"
    "\n"
    'Try: "Janet has 16 chickens. Each lays 3 eggs per day.\n'
    "She eats 2 and bakes 4 into muffins. She sells the\n"
    'rest at $2 each. How much does she earn per day?"'
)


def main():
    parser = argparse.ArgumentParser(description="Interactive CoDi chat")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="Skip latent reasoning steps (decode directly after encoding)")
    parser.add_argument("--oracle", action="store_true",
                        help="Run Latent Oracle to explain reasoning after each response")
    parser.add_argument("--oracle-dir", default="oracle_model/best",
                        help="Path to oracle adapter weights")
    parser.add_argument("--oracle-base", default=None,
                        help="Base model for oracle")
    args = parser.parse_args()

    console.print(Panel(
        BANNER_BODY,
        border_style="rgb(100,100,100)",
        box=box.DOUBLE,
        width=62,
    ))

    if args.no_reasoning:
        console.print("[bold rgb(190,40,40)]--no-reasoning:[/] latent thinking disabled\n")

    with console.status("[bold rgb(0,80,180)]Loading model...", spinner="dots"):
        model, prj, tokenizer, bot_id, eot_id = load_model()

    oracle_model = None
    oracle_tok = None
    if args.oracle:
        with console.status("[bold rgb(150,30,150)]Loading oracle...", spinner="dots"):
            from oracle_inference import load_oracle, oracle_answer
            from generate_oracle_data import make_oracle_input
            oracle_model, oracle_tok = load_oracle(args.oracle_dir, args.oracle_base)
        console.print("[bold rgb(150,30,150)]Oracle loaded.[/]")

    console.print("[bold rgb(0,130,50)]Ready![/] Type a math question. (quit to exit)\n")

    while True:
        try:
            console.print("[rgb(180,120,0)]━[/]" * 80)
            user_input = console.input("[bold rgb(180,120,0)]You:[/] ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            generate_response(model, prj, tokenizer, bot_id, eot_id, user_input,
                              no_reasoning=args.no_reasoning)
            console.print()

            # Oracle explanation
            if oracle_model and not args.no_reasoning:
                with console.status("[bold rgb(150,30,150)]Oracle analyzing...", spinner="dots"):
                    trace = generate_with_traces(
                        model, prj, tokenizer, bot_id, eot_id, user_input,
                        collect_no_reasoning=False,
                    )
                    steps_data = [{
                        "step": s.step,
                        "top_k_tokens": s.top_k_tokens,
                        "top_k_probs": s.top_k_probs,
                        "entropy": s.entropy,
                        "norm": s.norm,
                        "sparsity": s.sparsity,
                        "cosine_to_prev": s.cosine_to_prev,
                    } for s in trace.steps]

                    oracle_queries = [
                        "Summarize what the latent reasoning is doing for this input.",
                        "Which latent step is most critical for the final output?",
                    ]
                    explanations = []
                    for q in oracle_queries:
                        oi = make_oracle_input(user_input, steps_data, trace.output, q)
                        ans = oracle_answer(oracle_model, oracle_tok, oi)
                        explanations.append((q, ans))

                oracle_body = "\n".join(
                    f"[bold]{q}[/]\n{a}\n" for q, a in explanations
                )
                console.print(Panel(
                    oracle_body,
                    title="[bold rgb(150,30,150)]🔮 Latent Oracle[/]",
                    border_style="rgb(150,30,150)",
                    box=box.ROUNDED,
                    width=80,
                ))
        except KeyboardInterrupt:
            console.print("\nGoodbye!")
            break
        except EOFError:
            console.print("\nGoodbye!")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
