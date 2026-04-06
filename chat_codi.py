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
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.align import Align
from rich.console import Group
from rich import box

console = Console()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_DIR = "/root/.cache/huggingface/hub/models--bcywinski--codi_llama1b-answer_only/snapshots/315ec5cb5f9a071f950edb8455ca5c9a5f59691e"
NUM_LATENT = 6          # matches training config
LORA_ALPHA = 32         # from train script
LORA_R = 128


def load_model():
    """Reconstruct LlamaForCausalLM from checkpoint, merging LoRA into base weights."""
    console.print("[dark_cyan]Loading checkpoint...[/]")
    state_dict = torch.load(f"{CKPT_DIR}/pytorch_model.bin", map_location="cpu", weights_only=False)

    embed_w = state_dict["codi.base_model.model.model.embed_tokens.weight"]
    vocab_size, hidden_size = embed_w.shape

    num_layers = 0
    while f"codi.base_model.model.model.layers.{num_layers}.input_layernorm.weight" in state_dict:
        num_layers += 1

    inter_size = state_dict["codi.base_model.model.model.layers.0.mlp.gate_proj.base_layer.weight"].shape[0]

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        num_hidden_layers=num_layers,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )

    console.print(f"[dark_cyan]Building LlamaForCausalLM: {num_layers}L, {hidden_size}d, vocab={vocab_size}[/]")
    model = LlamaForCausalLM(config).to(torch.bfloat16)

    # Map checkpoint keys → plain model keys, merging LoRA
    prefix = "codi.base_model.model."
    new_sd = {}
    for key in state_dict:
        if key.startswith("prj.") or not key.startswith(prefix):
            continue
        if "lora_A" in key or "lora_B" in key:
            continue
        plain_key = key[len(prefix):].replace(".base_layer.", ".")
        new_sd[plain_key] = state_dict[key]

    scale = LORA_ALPHA / LORA_R
    for a_key in [k for k in state_dict if "lora_A.default.weight" in k]:
        b_key = a_key.replace("lora_A.default.weight", "lora_B.default.weight")
        plain_key = a_key[len(prefix):].replace(".lora_A.default.weight", ".base_layer.weight").replace(".base_layer.", ".")
        A = state_dict[a_key].float()
        B = state_dict[b_key].float()
        if plain_key in new_sd:
            new_sd[plain_key] = (new_sd[plain_key].float() + (B @ A) * scale).to(torch.bfloat16)

    missing, _ = model.load_state_dict(new_sd, strict=False)
    console.print(f"[dark_green]Loaded model ({len(new_sd)} tensors, {len(missing)} missing)[/]")

    # Projection layer
    prj = nn.Sequential(
        nn.Dropout(0.0),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )
    prj.add_module("ln", nn.LayerNorm(hidden_size))
    prj_sd = {k[4:]: state_dict[k] for k in state_dict if k.startswith("prj.")}
    prj.load_state_dict(prj_sd)
    prj = prj.to(torch.bfloat16).to(device)

    model = model.to(device)
    model.eval()
    prj.eval()

    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "[PAD]"

    ori_vocab = vocab_size - 3
    bot_id = ori_vocab + 1
    eot_id = ori_vocab + 2

    return model, prj, tokenizer, bot_id, eot_id


# ── Visualization ─────────────────────────────────────────────────────────────

def top_token_projections(latent_embd, embed_weight, tokenizer, k=8):
    latent = latent_embd.squeeze().float()
    vocab_w = embed_weight[: embed_weight.shape[0] - 3].float()
    sims = F.cosine_similarity(latent.unsqueeze(0), vocab_w, dim=1)
    topk = torch.topk(sims, k)
    tokens = []
    for idx, score in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        if tok:
            tokens.append((tok, score))
    return tokens


def norm_bar(value, max_val, width=15):
    """Small inline bar chart."""
    frac = min(1.0, max(0.0, value / max_val))
    filled = int(frac * width)
    return "▓" * filled + "░" * (width - filled)


def make_latent_panel(step, latent_embd, prev_embd, embed_weight, tokenizer, elapsed):
    projections = top_token_projections(latent_embd, embed_weight, tokenizer)
    norm = latent_embd.squeeze().float().norm().item()

    # Drift from previous latent
    drift_line = ""
    if prev_embd is not None:
        sim = F.cosine_similarity(
            latent_embd.flatten().float(), prev_embd.flatten().float(), dim=0
        ).item()
        bar = norm_bar(sim, 1.0, 20)
        drift_line = f"  cos(prev): [rgb(30,100,160)]{bar}[/] [bold]{sim:.4f}[/]"

    # Vocab projection tokens with gradient coloring (dark on light bg)
    proj_parts = []
    if projections:
        max_s = projections[0][1]
        min_s = projections[-1][1] if len(projections) > 1 else max_s
        for tok, score in projections:
            t = (score - min_s) / (max_s - min_s + 1e-8)
            # Dark purple → dark blue gradient, readable on white
            r = int(100 - 60 * t)
            g = int(40 + 20 * t)
            b = int(140 + 40 * t)
            color = f"rgb({r},{g},{b})"
            proj_parts.append(f"[bold {color}]\"{tok}\"[/]={score:.3f}")

    # Activation stats
    flat = latent_embd.squeeze().float()
    sparsity = (flat.abs() < 0.01).float().mean().item()

    stats_line = (
        f"  ‖h‖=[bold rgb(180,120,0)]{norm:.1f}[/]  "
        f"sparsity=[bold rgb(180,120,0)]{sparsity:.1%}[/]  "
        f"latency=[bold rgb(0,120,60)]{elapsed*1000:.0f}ms[/]"
    )
    body = stats_line + "\n"
    if drift_line:
        body += drift_line + "\n"
    body += f"  vocab projection: {' '.join(proj_parts)}"

    # Saturated dark colors that pop on white/light backgrounds
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
