#!/usr/bin/env python3
"""
Run the Latent Oracle to answer questions about CoDI reasoning traces.

Usage:
    # Interactive: ask questions about a prompt
    python oracle_inference.py --prompt "Janet's ducks lay 16 eggs..."

    # Batch: answer predefined questions for all traces
    python oracle_inference.py --traces traces_test.jsonl --output oracle_answers.jsonl

    # Custom query
    python oracle_inference.py --prompt "..." --query "Which step is most critical?"
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from codi_model import (
    load_model as load_codi, generate_with_traces, NUM_LATENT, device,
)
from generate_oracle_data import format_latent_trace, make_oracle_input


DEFAULT_QUERIES = [
    "Summarize what the latent reasoning is doing for this input.",
    "Which latent step is most critical for the final output?",
    "How does model confidence evolve across latent steps?",
    "At which step does the model first produce its final output?",
    "Does latent reasoning change the output for this input?",
    "Are any latent steps redundant?",
]


def load_oracle(model_dir, base_model=None):
    """Load the fine-tuned oracle model."""
    model_dir = Path(model_dir)

    # Try to find base model name from training config
    config_path = model_dir.parent / "train_config.json"
    if base_model is None and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        base_model = config.get("base", "HuggingFaceTB/SmolLM-360M-Instruct")

    if base_model is None:
        base_model = "HuggingFaceTB/SmolLM-360M-Instruct"

    print(f"Loading oracle: base={base_model}, adapter={model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, model_dir)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def oracle_answer(oracle_model, oracle_tokenizer, prompt_text, max_new_tokens=256):
    """Generate an oracle response given a formatted prompt."""
    inputs = oracle_tokenizer(prompt_text, return_tensors="pt", truncation=True,
                              max_length=768).to(device)
    with torch.no_grad():
        outputs = oracle_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=oracle_tokenizer.pad_token_id,
        )
    # Decode only the generated portion
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return oracle_tokenizer.decode(gen_ids, skip_special_tokens=True)


def interactive_mode(args):
    """Run CoDI on a prompt, then ask oracle questions interactively."""
    # Load CoDI
    print("Loading CoDI model...")
    model, prj, tokenizer, bot_id, eot_id = load_codi()

    # Load Oracle
    print("Loading oracle model...")
    oracle_model, oracle_tok = load_oracle(args.oracle_dir, args.oracle_base)

    # Run CoDI trace
    print(f"\nRunning CoDI on: {args.prompt[:80]}...")
    trace = generate_with_traces(model, prj, tokenizer, bot_id, eot_id, args.prompt)
    print(f"CoDI output: {trace.output}")
    if trace.no_reasoning_output:
        print(f"No-reasoning: {trace.no_reasoning_output}")

    steps_data = []
    for s in trace.steps:
        steps_data.append({
            "step": s.step,
            "top_k_tokens": s.top_k_tokens,
            "top_k_probs": s.top_k_probs,
            "entropy": s.entropy,
            "norm": s.norm,
            "sparsity": s.sparsity,
            "cosine_to_prev": s.cosine_to_prev,
        })

    # Answer queries
    queries = [args.query] if args.query else DEFAULT_QUERIES
    print(f"\n{'='*60}")
    for q in queries:
        oracle_input = make_oracle_input(args.prompt, steps_data, trace.output, q)
        answer = oracle_answer(oracle_model, oracle_tok, oracle_input)
        print(f"\nQ: {q}")
        print(f"A: {answer}")
    print(f"{'='*60}")

    # Interactive follow-up
    if not args.query:
        print("\nAsk more questions (or 'quit' to exit):")
        while True:
            try:
                q = input("Q: ").strip()
                if not q or q.lower() in ("quit", "exit", "q"):
                    break
                oracle_input = make_oracle_input(
                    args.prompt, steps_data, trace.output, q
                )
                answer = oracle_answer(oracle_model, oracle_tok, oracle_input)
                print(f"A: {answer}\n")
            except (KeyboardInterrupt, EOFError):
                break


def batch_mode(args):
    """Run oracle on pre-collected traces."""
    # Load Oracle only (traces already have CoDI data)
    print("Loading oracle model...")
    oracle_model, oracle_tok = load_oracle(args.oracle_dir, args.oracle_base)

    # Load traces
    print(f"Loading traces from {args.traces}...")
    records = []
    with open(args.traces) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    queries = DEFAULT_QUERIES
    output_path = Path(args.output or "oracle_answers.jsonl")

    with open(output_path, "w") as f:
        for i, record in enumerate(records):
            answers = {}
            for q in queries:
                oracle_input = make_oracle_input(
                    record["question"], record["steps"], record["output"], q
                )
                ans = oracle_answer(oracle_model, oracle_tok, oracle_input)
                answers[q] = ans

            result = {
                "index": record["index"],
                "question": record["question"],
                "codi_output": record["output"],
                "oracle_answers": answers,
            }
            f.write(json.dumps(result) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(records)}]")

    print(f"Done. Oracle answers saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Latent Oracle Inference")
    parser.add_argument("--oracle-dir", default="oracle_model/best",
                        help="Path to oracle adapter weights")
    parser.add_argument("--oracle-base", default=None,
                        help="Base model for oracle (auto-detected from config)")
    parser.add_argument("--prompt", default=None,
                        help="Run CoDI + oracle on this prompt")
    parser.add_argument("--query", default=None,
                        help="Specific question to ask the oracle")
    parser.add_argument("--traces", default=None,
                        help="Batch mode: run oracle on pre-collected traces")
    parser.add_argument("--output", default=None,
                        help="Batch mode output path")
    args = parser.parse_args()

    if args.traces:
        batch_mode(args)
    elif args.prompt:
        interactive_mode(args)
    else:
        parser.print_help()
        print("\nProvide --prompt for interactive mode or --traces for batch mode.")


if __name__ == "__main__":
    main()
