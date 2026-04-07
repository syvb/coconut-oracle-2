#!/usr/bin/env python3
"""
Fine-tune a small LM as a Latent Oracle that answers questions about
CoDI latent reasoning tokens.

Uses LoRA on a small decoder-only model (SmolLM-360M or TinyLlama-1.1B).

Usage:
    python train_oracle.py --train oracle_data.jsonl --val oracle_val.jsonl
    python train_oracle.py --base HuggingFaceTB/SmolLM-360M-Instruct --epochs 5
    python train_oracle.py --base TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


class OracleDataset(Dataset):
    """QA pairs formatted as input+response for causal LM training.

    Pre-tokenizes everything in batch during __init__ for speed.
    """

    def __init__(self, path, tokenizer, max_length=1024):
        self.max_length = max_length

        # Check for cached tokenized data
        cache_path = Path(path).with_suffix(".cache.pt")
        if cache_path.exists():
            print(f"    Loading cached tokenized data from {cache_path}")
            cached = torch.load(cache_path, weights_only=True)
            self.input_ids = cached["input_ids"]
            self.attention_mask = cached["attention_mask"]
            self.labels = cached["labels"]
            print(f"    Loaded {len(self)} samples from cache")
            return

        # Read all records
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line))

        n = len(records)
        print(f"    Tokenizing {n} samples (one-time, will cache)...")

        all_input_ids = []
        all_attention = []
        all_labels = []

        for i, rec in enumerate(records):
            text = rec["input"] + rec["response"] + tokenizer.eos_token
            enc = tokenizer(text, truncation=True, max_length=max_length,
                            padding="max_length")
            input_enc = tokenizer(rec["input"], add_special_tokens=False)
            input_len = len(input_enc["input_ids"])

            ids = torch.tensor(enc["input_ids"], dtype=torch.long)
            mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
            lab = ids.clone()
            lab[:input_len] = -100
            lab[mask == 0] = -100

            all_input_ids.append(ids)
            all_attention.append(mask)
            all_labels.append(lab)

            if (i + 1) % 5000 == 0 or (i + 1) == n:
                print(f"      {i+1}/{n}")

        self.input_ids = torch.stack(all_input_ids)
        self.attention_mask = torch.stack(all_attention)
        self.labels = torch.stack(all_labels)

        # Save cache
        torch.save({
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }, cache_path)
        print(f"    Saved cache to {cache_path}")

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def main():
    parser = argparse.ArgumentParser(description="Train Latent Oracle")
    parser.add_argument("--train", required=True, help="Training data JSONL")
    parser.add_argument("--val", default=None, help="Validation data JSONL")
    parser.add_argument("--base", default="HuggingFaceTB/SmolLM-360M-Instruct",
                        help="Base model")
    parser.add_argument("--output-dir", default="oracle_model",
                        help="Output directory for adapter weights")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load tokenizer and model
    print(f"Loading base model: {args.base}")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # Load data
    print(f"Loading training data from {args.train}")
    train_ds = OracleDataset(args.train, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)
    print(f"  {len(train_ds)} training samples, {len(train_loader)} batches/epoch")

    val_loader = None
    if args.val:
        val_ds = OracleDataset(args.val, tokenizer, args.max_length)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        print(f"  {len(val_ds)} validation samples")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs ({total_steps} optimizer steps)")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += outputs.loss.item()
            num_batches += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    print(f"  epoch {epoch+1} step {global_step}  "
                          f"loss={avg:.4f}  lr={lr:.2e}")

        avg_train = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={avg_train:.4f}")

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    labels=labels)
                    val_loss += outputs.loss.item()
                    val_batches += 1
            avg_val = val_loss / max(val_batches, 1)
            print(f"  val_loss={avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                model.save_pretrained(output_dir / "best")
                tokenizer.save_pretrained(output_dir / "best")
                print(f"  -> saved best model (val_loss={avg_val:.4f})")

        # Save checkpoint each epoch
        model.save_pretrained(output_dir / f"epoch{epoch+1}")
        tokenizer.save_pretrained(output_dir / f"epoch{epoch+1}")

    # Save final
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    print(f"\nTraining complete. Models saved to {output_dir}/")

    # Save config for reference
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
