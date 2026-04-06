"""
Shared CoDi model loading and inference.

Reconstructs a LlamaForCausalLM from the bcywinski/codi_llama1b-answer_only
checkpoint by merging LoRA weights into the base model.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

CKPT_DIR = "/root/.cache/huggingface/hub/models--bcywinski--codi_llama1b-answer_only/snapshots/315ec5cb5f9a071f950edb8455ca5c9a5f59691e"
NUM_LATENT = 6
LORA_ALPHA = 32
LORA_R = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(verbose=True):
    """Load CoDi model, returning (model, prj, tokenizer, bot_id, eot_id)."""
    if verbose:
        print("Loading checkpoint...")
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

    if verbose:
        print(f"Building LlamaForCausalLM: {num_layers}L, {hidden_size}d, vocab={vocab_size}")
    model = LlamaForCausalLM(config).to(torch.bfloat16)

    # Map checkpoint keys to plain model keys, merging LoRA
    prefix = "codi.base_model.model."
    new_sd = {}
    for key in state_dict:
        if key.startswith("prj.") or not key.startswith(prefix):
            continue
        if "lora_A" in key or "lora_B" in key:
            continue
        new_sd[key[len(prefix):].replace(".base_layer.", ".")] = state_dict[key]

    scale = LORA_ALPHA / LORA_R
    for a_key in [k for k in state_dict if "lora_A.default.weight" in k]:
        b_key = a_key.replace("lora_A.default.weight", "lora_B.default.weight")
        plain_key = (
            a_key[len(prefix):]
            .replace(".lora_A.default.weight", ".base_layer.weight")
            .replace(".base_layer.", ".")
        )
        A = state_dict[a_key].float()
        B = state_dict[b_key].float()
        if plain_key in new_sd:
            new_sd[plain_key] = (new_sd[plain_key].float() + (B @ A) * scale).to(torch.bfloat16)

    model.load_state_dict(new_sd, strict=False)

    # Projection layer
    prj = nn.Sequential(
        nn.Dropout(0.0),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )
    prj.add_module("ln", nn.LayerNorm(hidden_size))
    prj.load_state_dict({k[4:]: state_dict[k] for k in state_dict if k.startswith("prj.")})
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

    if verbose:
        print("Model loaded.")

    return model, prj, tokenizer, bot_id, eot_id


@torch.no_grad()
def generate(model, prj, tokenizer, bot_id, eot_id, prompt,
             max_new_tokens=256, greedy=True, num_latent=NUM_LATENT):
    """Run CoDi inference and return the decoded string.

    Set num_latent=0 to skip latent reasoning (no-reasoning mode).
    """
    embed_fn = model.model.embed_tokens
    ori_vocab = model.config.vocab_size - 3

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    bot_tensor = torch.tensor([[bot_id]], dtype=torch.long, device=device)
    input_ids = torch.cat([inputs["input_ids"], bot_tensor], dim=1)

    # Encode
    outputs = model(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    past_kv = outputs.past_key_values
    latent_embd = prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))

    # Latent reasoning
    for _ in range(num_latent):
        outputs = model(
            inputs_embeds=latent_embd, use_cache=True,
            output_hidden_states=True, past_key_values=past_kv,
        )
        past_kv = outputs.past_key_values
        latent_embd = prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))

    # Decode
    eot_emb = embed_fn(torch.tensor([[eot_id]], dtype=torch.long, device=device))
    output_emb = eot_emb
    generated_tokens = []

    for _ in range(max_new_tokens):
        out = model(inputs_embeds=output_emb, use_cache=True, past_key_values=past_kv)
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :ori_vocab]
        next_id = torch.argmax(logits, dim=-1) if greedy else None
        if next_id is None:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        tok_id = next_id.item()
        if tok_id == tokenizer.eos_token_id or tok_id >= ori_vocab:
            break
        generated_tokens.append(tok_id)
        output_emb = embed_fn(next_id.unsqueeze(0))

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def extract_number(text):
    """Extract the last number from a string."""
    text = text.replace(",", "")
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return float("inf")
    return float(nums[-1])
