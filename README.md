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

## Results (full GSM8K test set, 1319 questions)

| Mode | Accuracy | Throughput | Time |
|------|----------|------------|------|
| Reasoning (6 latent steps) | **36.7%** | 7.0 q/s | 188s |
| No reasoning (0 steps) | 28.7% | 7.7 q/s | 171s |

Reasoning advantage: **+8.0%** (helped 134 questions, hurt 29).

## How CoDi inference works

1. **Encode**: Tokenize the question, append `<bot>` token, run through the model
2. **Think**: Take the last hidden state, project through the MLP, feed back as input — repeat 6 times (no text generated, just latent vectors cycling through the transformer)
3. **Decode**: Append `<eot>` token, then autoregressively generate the answer

The `--no-reasoning` flag skips step 2, going straight from encoding to decoding.

## Latent Oracle: Interpreting CoDI's Reasoning Tokens

Inspired by [Activation Oracles](https://arxiv.org/abs/2512.15674), we train a small LM (SmolLM-360M) to answer natural language questions about what CoDI's latent reasoning tokens are doing. All training data is generated via **task-agnostic unsupervised methods** — no knowledge of the downstream task (GSM8K math) is assumed.

### Training Data Generation (5 strategies, all task-agnostic)

| Strategy | Method | Yield |
|----------|--------|-------|
| **Perturbation/Ablation** | Zero out or skip individual latent steps, observe output changes | 11,148 |
| **Early Decoding** | Fork KV cache after each step, decode partial answer — reveals what model "knows" at each point | 11,871 |
| **Contrastive** | Compare reasoning-on (6 steps) vs reasoning-off (0 steps) outputs | 1,866 |
| **Token Prediction Stats** | Project hidden states to vocab space via `lm_head`; extract entropy, top-k tokens, convergence | 26,380 |
| **Summary** | Composite questions combining signals from all strategies | 1,319 |
| **Total** | | **52,584** |

### Oracle Training

- **Base model**: [SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-360M-Instruct)
- **Fine-tuning**: LoRA (r=32, alpha=64) on all attention + MLP modules
- **Data**: 47,326 train / 5,258 val QA pairs
- **Input format**: Structured text encoding the question, per-step top-k tokens/probs/entropy, and model output

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.1146 | 0.0488 |
| 2 | 0.0432 | 0.0352 |
| 3 | 0.0283 | 0.0307 |

### Oracle Evaluation (100 held-out questions)

| Metric | Accuracy | What it measures |
|--------|----------|-----------------|
| Ablation prediction | **76%** | Can oracle identify which step is most critical? |
| Convergence prediction | **72%** | Can oracle predict when output stabilizes? (±1 step) |
| Contrastive | **73%** | Can oracle predict whether reasoning changes output? |
| Redundancy consistency | **88%** | Do oracle's redundancy claims match actual ablation data? |

### Oracle Scripts

```bash
# 1. Collect traces (latent states + ablations + early decodes)
python collect_traces.py -n 1319 --output traces_test.jsonl

# 2. Generate task-agnostic QA training data
python generate_oracle_data.py --input traces_test.jsonl

# 3. Train oracle (SmolLM-360M + LoRA)
python train_oracle.py --train oracle_data.jsonl --val oracle_val.jsonl

# 4. Evaluate
python eval_oracle.py --traces traces_test.jsonl --oracle-dir oracle_model/best

# 5. Interactive: ask questions about CoDI's reasoning
python oracle_inference.py --prompt "Janet has 16 ducks..."

# 6. Integrated chat with oracle explanations
python chat_codi.py --oracle
```

### Key Findings

- **Early decoding is the most revealing strategy**: partial answers show progressive refinement (e.g., pre-reasoning outputs "36", after step 2 corrects to "18")
- **CoDI is robust to individual step ablation**: for many questions, zeroing a single step doesn't change the output, suggesting distributed/redundant representations
- **The oracle achieves 88% consistency on redundancy claims** against ground-truth ablation data, demonstrating that the text-based trace representation (top-k tokens + entropy) captures meaningful information about latent reasoning

## Reference

- [CoDi paper](https://arxiv.org/abs/2502.21074) (EMNLP 2025)
- [Activation Oracles paper](https://arxiv.org/abs/2512.15674)
- [Official CoDi repo](https://github.com/zhenyi4/codi) (cloned at `/workspace/codi`)
