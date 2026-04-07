#!/usr/bin/env python3
"""
Generate task-agnostic oracle training data from collected traces.

Reads JSONL traces (from collect_traces.py) and produces QA pairs using
six strategies that require no knowledge of the downstream task:

1. Perturbation/Ablation - what happens when steps are removed/modified
2. Early Decoding - what does the model know at each step
3. Contrastive - reasoning vs no-reasoning comparison
4. Counterfactual Input Sensitivity - which steps change with input edits
5. Token Prediction Statistics - confidence, entropy, convergence
6. Attention / Information Flow - (requires attention data, optional)

Usage:
    python generate_oracle_data.py --input traces_test.jsonl
    python generate_oracle_data.py --input traces_test.jsonl --val-fraction 0.1
"""

import argparse
import json
import random
from pathlib import Path

from codi_model import extract_number, NUM_LATENT


def format_latent_trace(steps):
    """Format latent step data as structured text for the oracle input."""
    lines = []
    for s in steps:
        tokens_str = ", ".join(repr(t) for t in s["top_k_tokens"][:5])
        probs_str = ", ".join(f"{p:.3f}" for p in s["top_k_probs"][:5])
        cos_str = f"  cos_prev={s['cosine_to_prev']:.4f}" if s["cosine_to_prev"] is not None else ""
        lines.append(
            f"Step {s['step']}: top=[{tokens_str}] probs=[{probs_str}] "
            f"entropy={s['entropy']:.2f} norm={s['norm']:.1f} "
            f"sparsity={s['sparsity']:.1%}{cos_str}"
        )
    return "\n".join(lines)


def make_oracle_input(question, steps, output, query):
    """Build the full oracle input prompt."""
    trace_text = format_latent_trace(steps)
    return (
        f"<input>{question}</input>\n"
        f"<latent_trace>\n{trace_text}\n</latent_trace>\n"
        f"<model_output>{output}</model_output>\n"
        f"<query>{query}</query>\n"
        f"<response>"
    )


# ── Strategy 1: Ablation QA ─────────────────────────────────────────────────

def generate_ablation_qa(record):
    """Generate QA pairs from ablation experiments."""
    pairs = []
    if "ablations" not in record:
        return pairs

    original = record["output"]
    steps = record["steps"]
    question = record["question"]
    ablations = record["ablations"]

    # Per-step ablation effects
    for step_idx in range(NUM_LATENT):
        zero_key = f"zero_step{step_idx}"
        if zero_key not in ablations:
            continue
        ablated_output = ablations[zero_key]
        changed = ablated_output.strip() != original.strip()

        # Q: What happens if step N is removed?
        if changed:
            response = (
                f"The output changes. Original: {repr(original.strip())}. "
                f"Without step {step_idx}: {repr(ablated_output.strip())}."
            )
        else:
            response = (
                f"No change. The output remains {repr(original.strip())} "
                f"even without step {step_idx}."
            )

        pairs.append({
            "input": make_oracle_input(
                question, steps, original,
                f"What happens if latent step {step_idx} is removed?"
            ),
            "response": response,
            "strategy": "ablation",
            "sub_type": "step_effect",
        })

    # Q: Which step is most critical?
    max_change = -1
    critical_step = 0
    for step_idx in range(NUM_LATENT):
        zero_key = f"zero_step{step_idx}"
        if zero_key not in ablations:
            continue
        ablated = ablations[zero_key].strip()
        # Use character-level edit distance as a simple change metric
        change = sum(1 for a, b in zip(original, ablated) if a != b) + abs(len(original) - len(ablated))
        if change > max_change:
            max_change = change
            critical_step = step_idx

    # Count redundant steps
    redundant = []
    for step_idx in range(NUM_LATENT):
        zero_key = f"zero_step{step_idx}"
        if zero_key in ablations and ablations[zero_key].strip() == original.strip():
            redundant.append(step_idx)

    pairs.append({
        "input": make_oracle_input(
            question, steps, original,
            "Which latent step is most critical for the final output?"
        ),
        "response": (
            f"Step {critical_step} is most critical -- removing it causes the largest "
            f"output change (edit distance {max_change})."
        ),
        "strategy": "ablation",
        "sub_type": "critical_step",
    })

    if redundant:
        pairs.append({
            "input": make_oracle_input(
                question, steps, original,
                "Are any latent steps redundant?"
            ),
            "response": (
                f"Steps {redundant} can be removed without changing the output."
            ),
            "strategy": "ablation",
            "sub_type": "redundant_steps",
        })

    # Noise vs zero comparison
    for step_idx in range(NUM_LATENT):
        zero_key = f"zero_step{step_idx}"
        noise_key = f"noise_step{step_idx}"
        skip_key = f"skip_step{step_idx}"
        if zero_key not in ablations:
            continue

        zero_out = ablations[zero_key].strip()
        noise_out = ablations.get(noise_key, "").strip()
        skip_out = ablations.get(skip_key, "").strip()

        effects = []
        if zero_out != original.strip():
            effects.append(f"zeroing -> {repr(zero_out)}")
        if noise_out and noise_out != original.strip():
            effects.append(f"noise -> {repr(noise_out)}")
        if skip_out and skip_out != original.strip():
            effects.append(f"skip -> {repr(skip_out)}")

        if len(effects) >= 2:
            pairs.append({
                "input": make_oracle_input(
                    question, steps, original,
                    f"How do different ablation types affect step {step_idx}?"
                ),
                "response": (
                    f"Original output: {repr(original.strip())}. "
                    f"Effects: {'; '.join(effects)}."
                ),
                "strategy": "ablation",
                "sub_type": "ablation_comparison",
            })

    return pairs


# ── Strategy 2: Early Decoding QA ────────────────────────────────────────────

def generate_early_decode_qa(record):
    """Generate QA from progressive early decoding."""
    pairs = []
    if "early_decodes" not in record:
        return pairs

    question = record["question"]
    steps = record["steps"]
    final_output = record["output"]
    early = record["early_decodes"]

    # Per-step: what does the model output at this point?
    for ed in early:
        after_step = ed["after_step"]
        partial = ed["output"]
        label = f"step {after_step}" if after_step >= 0 else "encoding (before any latent steps)"

        pairs.append({
            "input": make_oracle_input(
                question, steps, final_output,
                f"What does the model output after {label}?"
            ),
            "response": (
                f"After {label}, the model outputs: {repr(partial.strip())}."
            ),
            "strategy": "early_decode",
            "sub_type": "step_output",
        })

    # At which step does the output first match the final answer?
    first_match = None
    for ed in early:
        if ed["output"].strip() == final_output.strip():
            first_match = ed["after_step"]
            break

    if first_match is not None:
        label = f"step {first_match}" if first_match >= 0 else "encoding"
        pairs.append({
            "input": make_oracle_input(
                question, steps, final_output,
                "At which step does the model first produce its final output?"
            ),
            "response": (
                f"The model first produces the final output ({repr(final_output.strip())}) "
                f"after {label}. "
                f"{'All subsequent steps produce the same output.' if first_match < NUM_LATENT - 1 else ''}"
            ),
            "strategy": "early_decode",
            "sub_type": "convergence_point",
        })
    else:
        pairs.append({
            "input": make_oracle_input(
                question, steps, final_output,
                "At which step does the model first produce its final output?"
            ),
            "response": (
                "The output is still changing at the final step, suggesting the model "
                "has not fully converged within the available latent steps."
            ),
            "strategy": "early_decode",
            "sub_type": "convergence_point",
        })

    # Evolution narrative
    if len(early) >= 3:
        evolution = []
        for ed in early:
            label = f"step {ed['after_step']}" if ed["after_step"] >= 0 else "pre-reasoning"
            evolution.append(f"{label}: {repr(ed['output'].strip())}")
        pairs.append({
            "input": make_oracle_input(
                question, steps, final_output,
                "How does the model's output evolve across latent steps?"
            ),
            "response": "Progressive outputs: " + " -> ".join(evolution),
            "strategy": "early_decode",
            "sub_type": "evolution",
        })

    return pairs


# ── Strategy 3: Contrastive (reasoning vs no-reasoning) ─────────────────────

def generate_contrastive_qa(record):
    """Generate QA comparing reasoning-on vs reasoning-off."""
    pairs = []
    if record.get("no_reasoning_output") is None:
        return pairs

    question = record["question"]
    steps = record["steps"]
    with_reasoning = record["output"].strip()
    without_reasoning = record["no_reasoning_output"].strip()
    changed = with_reasoning != without_reasoning

    pairs.append({
        "input": make_oracle_input(
            question, steps, with_reasoning,
            "Does latent reasoning change the output for this input?"
        ),
        "response": (
            f"{'Yes' if changed else 'No'}. "
            f"Without reasoning: {repr(without_reasoning)}. "
            f"With reasoning: {repr(with_reasoning)}."
        ),
        "strategy": "contrastive",
        "sub_type": "reasoning_effect",
    })

    if changed:
        # Edit distance as proxy for how much it changed
        edit_dist = sum(1 for a, b in zip(with_reasoning, without_reasoning) if a != b) + \
                    abs(len(with_reasoning) - len(without_reasoning))
        pairs.append({
            "input": make_oracle_input(
                question, steps, with_reasoning,
                "How much does latent reasoning change the output?"
            ),
            "response": (
                f"The output changes significantly (edit distance {edit_dist}). "
                f"Without reasoning: {repr(without_reasoning)}. "
                f"With reasoning: {repr(with_reasoning)}."
            ),
            "strategy": "contrastive",
            "sub_type": "reasoning_magnitude",
        })

    return pairs


# ── Strategy 5: Token Prediction Statistics ──────────────────────────────────

def generate_token_stats_qa(record):
    """Generate QA about confidence, entropy, and convergence."""
    pairs = []
    question = record["question"]
    steps = record["steps"]
    output = record["output"]

    # Per-step confidence
    for s in steps:
        top_tok = s["top_k_tokens"][0] if s["top_k_tokens"] else "?"
        top_prob = s["top_k_probs"][0] if s["top_k_probs"] else 0.0
        pairs.append({
            "input": make_oracle_input(
                question, steps, output,
                f"How confident is the model at latent step {s['step']}?"
            ),
            "response": (
                f"Entropy is {s['entropy']:.2f}. "
                f"Top token is {repr(top_tok)} with probability {top_prob:.3f}. "
                f"{'High' if s['entropy'] < 2.0 else 'Moderate' if s['entropy'] < 4.0 else 'Low'} "
                f"confidence."
            ),
            "strategy": "token_stats",
            "sub_type": "step_confidence",
        })

    # What tokens does the model consider at step N?
    for s in steps:
        tokens_str = ", ".join(
            f"{repr(t)} ({p:.3f})"
            for t, p in zip(s["top_k_tokens"][:5], s["top_k_probs"][:5])
        )
        pairs.append({
            "input": make_oracle_input(
                question, steps, output,
                f"What tokens does the model consider most likely at step {s['step']}?"
            ),
            "response": f"Top 5 tokens: {tokens_str}.",
            "strategy": "token_stats",
            "sub_type": "top_tokens",
        })

    # Entropy trajectory
    entropies = [s["entropy"] for s in steps]
    min_idx = entropies.index(min(entropies))
    max_idx = entropies.index(max(entropies))
    trajectory = ", ".join(f"step {s['step']}: {s['entropy']:.2f}" for s in steps)
    pairs.append({
        "input": make_oracle_input(
            question, steps, output,
            "How does model confidence evolve across latent steps?"
        ),
        "response": (
            f"Entropy trajectory: {trajectory}. "
            f"Most confident at step {min_idx} (entropy {entropies[min_idx]:.2f}), "
            f"least confident at step {max_idx} (entropy {entropies[max_idx]:.2f})."
        ),
        "strategy": "token_stats",
        "sub_type": "entropy_trajectory",
    })

    # KL-like divergence between consecutive steps (using top-k overlap as proxy)
    for i in range(1, len(steps)):
        prev_set = set(steps[i-1]["top_k_tokens"][:5])
        curr_set = set(steps[i]["top_k_tokens"][:5])
        overlap = len(prev_set & curr_set)
        pairs.append({
            "input": make_oracle_input(
                question, steps, output,
                f"How much does the token distribution change between steps {i-1} and {i}?"
            ),
            "response": (
                f"Top-5 token overlap: {overlap}/5. "
                f"Cosine similarity: {steps[i]['cosine_to_prev']:.4f}. "
                f"{'Stable' if overlap >= 4 and steps[i]['cosine_to_prev'] > 0.95 else 'Shifting'}."
            ),
            "strategy": "token_stats",
            "sub_type": "step_divergence",
        })

    # Convergence: does top-1 stabilize?
    top1_tokens = [s["top_k_tokens"][0] for s in steps]
    stable_from = None
    for i in range(len(top1_tokens)):
        if all(t == top1_tokens[i] for t in top1_tokens[i:]):
            stable_from = i
            break
    if stable_from is not None:
        pairs.append({
            "input": make_oracle_input(
                question, steps, output,
                "At which step does the model's top prediction stabilize?"
            ),
            "response": (
                f"Top-1 token stabilizes at step {stable_from} "
                f"(token: {repr(top1_tokens[stable_from])}). "
                f"All subsequent steps predict the same token."
            ),
            "strategy": "token_stats",
            "sub_type": "stabilization",
        })

    # Norm trajectory
    norms = [s["norm"] for s in steps]
    norm_str = ", ".join(f"step {s['step']}: {s['norm']:.1f}" for s in steps)
    pairs.append({
        "input": make_oracle_input(
            question, steps, output,
            "How does the hidden state magnitude change across steps?"
        ),
        "response": (
            f"Norms: {norm_str}. "
            f"{'Increasing' if norms[-1] > norms[0] * 1.1 else 'Decreasing' if norms[-1] < norms[0] * 0.9 else 'Stable'} "
            f"trend."
        ),
        "strategy": "token_stats",
        "sub_type": "norm_trajectory",
    })

    return pairs


# ── Strategy composite: Overall summary ──────────────────────────────────────

def generate_summary_qa(record):
    """Generate overall reasoning summary questions."""
    pairs = []
    question = record["question"]
    steps = record["steps"]
    output = record["output"]

    # Combine multiple signals into a summary
    entropies = [s["entropy"] for s in steps]
    top1s = [s["top_k_tokens"][0] for s in steps]

    # Find convergence
    early = record.get("early_decodes", [])
    evolution_parts = []
    for ed in early:
        label = f"step {ed['after_step']}" if ed["after_step"] >= 0 else "pre"
        evolution_parts.append(f"{label}={repr(ed['output'].strip()[:50])}")

    # Critical step from ablation
    ablations = record.get("ablations", {})
    critical_info = ""
    if ablations:
        max_change = -1
        crit = 0
        for step_idx in range(NUM_LATENT):
            key = f"zero_step{step_idx}"
            if key in ablations:
                abl = ablations[key].strip()
                change = sum(1 for a, b in zip(output, abl) if a != b) + abs(len(output) - len(abl))
                if change > max_change:
                    max_change = change
                    crit = step_idx
        critical_info = f" Step {crit} is most critical (ablation causes largest change)."

    no_reason = record.get("no_reasoning_output", "")
    reason_diff = ""
    if no_reason is not None:
        if no_reason.strip() != output.strip():
            reason_diff = f" Reasoning changes output from {repr(no_reason.strip()[:50])} to {repr(output.strip()[:50])}."
        else:
            reason_diff = " Reasoning does not change the output."

    evolution_str = " -> ".join(evolution_parts[:4]) if evolution_parts else "N/A"

    pairs.append({
        "input": make_oracle_input(
            question, steps, output,
            "Summarize what the latent reasoning is doing for this input."
        ),
        "response": (
            f"The model processes this input through {NUM_LATENT} latent steps. "
            f"Top-1 tokens across steps: {top1s}. "
            f"Entropy range: {min(entropies):.2f} to {max(entropies):.2f}. "
            f"Early decode evolution: {evolution_str}."
            f"{critical_info}{reason_diff}"
        ),
        "strategy": "summary",
        "sub_type": "overall",
    })

    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate oracle training data")
    parser.add_argument("--input", required=True, help="Input traces JSONL")
    parser.add_argument("--output", default=None,
                        help="Output JSONL (default: oracle_data.jsonl)")
    parser.add_argument("--val-output", default=None,
                        help="Validation output (default: oracle_val.jsonl)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output or "oracle_data.jsonl")
    val_path = Path(args.val_output or "oracle_val.jsonl")

    random.seed(args.seed)

    # Load traces
    print(f"Loading traces from {args.input}...")
    records = []
    with open(args.input) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} trace records")

    # Generate QA pairs
    all_pairs = []
    strategy_counts = {}

    for record in records:
        generators = [
            generate_ablation_qa,
            generate_early_decode_qa,
            generate_contrastive_qa,
            generate_token_stats_qa,
            generate_summary_qa,
        ]
        for gen_fn in generators:
            pairs = gen_fn(record)
            for p in pairs:
                p["source_index"] = record["index"]
                strat = p["strategy"]
                strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
            all_pairs.extend(pairs)

    print(f"\nGenerated {len(all_pairs)} QA pairs:")
    for strat, count in sorted(strategy_counts.items()):
        print(f"  {strat:20s}: {count:6d}")

    # Shuffle and split
    random.shuffle(all_pairs)
    val_size = int(len(all_pairs) * args.val_fraction)
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    # Write
    with open(output_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"\nTrain: {len(train_pairs)} pairs -> {output_path}")

    with open(val_path, "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Val:   {len(val_pairs)} pairs -> {val_path}")


if __name__ == "__main__":
    main()
