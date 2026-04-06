# coconut-oracle-2

Experiments with [CoDi](https://arxiv.org/abs/2502.21074) (Compressing Chain-of-Thought into Continuous Space via Self-Distillation) using the [`bcywinski/codi_llama1b-answer_only`](https://huggingface.co/bcywinski/codi_llama1b-answer_only) checkpoint.

CoDi compresses explicit chain-of-thought reasoning into continuous latent vectors. Instead of generating reasoning tokens in natural language, the model "thinks" in embedding space through a series of latent steps, then decodes just the final answer.

## Model

- **Base**: LLaMA 3.2 1B Instruct
- **Fine-tuning**: LoRA (r=128, alpha=32) on all attention + MLP modules
- **Projection layer**: 2-layer MLP with GELU + LayerNorm, maps hidden states back to input space between latent steps
- **Latent steps**: 6 continuous reasoning tokens per inference
- **Training data**: GSM8K-Aug (augmented GSM8K)
- **Eval data**: GSM8K test split (1319 questions, not in training set)

The checkpoint stores base weights + LoRA weights separately. Our loader merges them at load time, so no gated HF access to the original LLaMA repo is needed.

## Scripts

### `chat_codi.py` — Interactive chat

```bash
python chat_codi.py               # with latent reasoning visualization
python chat_codi.py --no-reasoning  # skip latent steps for comparison
```

Each latent thinking step is visualized with:
- A heatmap of the 2048-dim hidden state (blue=negative, gray=zero, red=positive)
- A delta heatmap showing what changed from the previous step
- Cosine similarity drift, vector norm, sparsity stats

### `bench_codi.py` — GSM8K benchmark

```bash
python bench_codi.py                     # full test set, both modes
python bench_codi.py -n 100              # first 100 examples
python bench_codi.py --mode reasoning    # reasoning only
python bench_codi.py --mode none         # no-reasoning only
python bench_codi.py --output results.json  # save detailed results
```

### `codi_model.py` — Shared library

Model loading and inference used by both scripts. Key functions:
- `load_model()` — reconstruct LlamaForCausalLM with merged LoRA weights
- `generate(model, prj, tokenizer, bot_id, eot_id, prompt, num_latent=6)` — run inference (set `num_latent=0` to skip reasoning)
- `extract_number(text)` — pull the last number from model output

## Results (100 examples)

| Mode | Accuracy | Throughput |
|------|----------|------------|
| Reasoning (6 latent steps) | **39.0%** | 7.2 q/s |
| No reasoning (0 steps) | 36.0% | 9.4 q/s |

Reasoning advantage: **+3.0%** (helped 6 questions, hurt 3). Full 1319-question benchmark pending.

## How CoDi inference works

1. **Encode**: Tokenize the question, append `<bot>` token, run through the model
2. **Think**: Take the last hidden state, project through the MLP, feed back as input — repeat 6 times (no text generated, just latent vectors cycling through the transformer)
3. **Decode**: Append `<eot>` token, then autoregressively generate the answer

The `--no-reasoning` flag skips step 2, going straight from encoding to decoding.

## Reference

- [CoDi paper](https://arxiv.org/abs/2502.21074) (EMNLP 2025)
- [Official CoDi repo](https://github.com/zhenyi4/codi) (cloned at `/workspace/codi`)
