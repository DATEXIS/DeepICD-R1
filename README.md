# DeepICD-R1

**Reinforcement Learning from Clinical Coding Feedback: Training LLMs to Reason about ICD-10 Diagnoses**

This repository contains the code for the paper *"DeepICD-R1"*, which applies R1-Zero-style group relative policy optimization (GRPO) to train large language models for prospective ICD-10 clinical coding. Given only an admission note, the model is trained to generate step-by-step reasoning traces that culminate in ICD-10 code predictions.

---

## Overview

Standard clinical coding assigns ICD-10 codes after discharge based on full documentation. We tackle a harder, more clinically meaningful problem: **prospective coding** — predicting the discharge codes from the admission note alone.

Our approach:
1. The model produces interleaved `<think>` / `<diagnosis>` pairs, one per predicted code.
2. A composite reward signal shapes training:
   - **Outcome reward**: partial-overlap matching of predicted codes against gold ICD-10 labels via a MeiliSearch index.
   - **Format reward**: reward for correctly structured reasoning traces.
   - **Length malus**: penalty proportional to output length to discourage verbosity.
   - **LLM-as-a-judge** (optional): a Llama-3 model evaluates reasoning quality against ICD-10 CM guidelines.
3. Training uses [verl](https://github.com/volcengine/verl) (primary) or [TRL](https://github.com/huggingface/trl) as the GRPO backend.

---

## Repository Structure

```
.
├── src/
│   ├── rewards.py                      # Reward functions (verl & TRL compatible)
│   ├── evaluate.py                     # Evaluation pipeline (vLLM inference + metrics)
│   ├── train_clinical_r1_zero.py       # TRL-based GRPO training entry point
│   ├── create_oracle_reasoning_traces.py  # SFT warm-start trace generation
│   ├── prompts/                        # Task prompts (CSV/JSON)
│   ├── setup/                          # Data preprocessing and MeiliSearch indexing
│   ├── sft/                            # Supervised fine-tuning (warm start)
│   └── utils/                          # Metrics, formatters, label helpers, IO
├── data/
│   ├── README.md                       # Data access instructions
│   └── icd_10_cm_guidelines/           # ICD-10-CM coding guidelines (public domain)
├── configs/                            # Accelerate / DeepSpeed / FSDP configs
├── k8s/train/                          # verl hyperparameter reference (Kubernetes jobs)
├── test/                               # Unit tests for reward functions
├── Dockerfile_train                    # Reproducible training environment
└── pyproject.toml
```

---

## Requirements

- Python ≥ 3.12
- CUDA-capable GPU(s) — experiments were run on 1–8 × H100 80GB
- [verl](https://github.com/volcengine/verl) with FlashAttention 2 (for primary training pipeline) — see `Dockerfile_train`
- [MeiliSearch](https://www.meilisearch.com/) instance with an ICD-10 index (see [Setup](#setup))
- [TRL](https://github.com/huggingface/trl) ≥ 0.12 (optional, for the TRL training path)

### Installation

```bash
git clone https://github.com/DATEXIS/DeepICD-R1
cd DeepICD-R1
pip install -e ".[evaluate,sft]"
```

For verl-based training, use the provided Docker image:

```bash
docker build -f Dockerfile_train -t deep-icd-r1-train .
```

---

## Data

This project uses **MIMIC-IV** clinical notes, which require credentialed access via [PhysioNet](https://physionet.org/content/mimiciv/). See [data/README.md](data/README.md) for instructions on obtaining access, preprocessing the notes, and building the required dataset splits.

---

## Setup

### 1. Preprocess data

Convert raw MIMIC parquet files into the prompt format expected by the training pipeline:

```bash
python src/setup/preprocess_data_verl.py \
    --data_paths <path-to-mimic-iv-parquets> \
    --prompt_file src/prompts/prospective_icd_coding.csv \
    --output_folder data/processed/
```

### 2. Start MeiliSearch and index ICD-10 codes

The reward function uses MeiliSearch for fuzzy ICD-10 code lookup at training time:

```bash
# Start MeiliSearch (Docker)
docker run -d -p 7700:7700 getmeili/meilisearch:latest

# Index ICD-10 codes
python src/setup/fill_meili.py \
    --icd_codes <path-to-icd_codes.csv> \
    --index mimic_icd_10_codes
```

---

## Training

### verl (recommended)

The `k8s/train/` directory contains reference job manifests with all hyperparameters used in the paper. To run locally:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/processed/train_prospective_icd_coding.parquet \
    data.val_files=data/processed/val_prospective_icd_coding.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    custom_reward_function.path=src/rewards.py \
    custom_reward_function.name=verl_batched_compute_score_multiple_think_traces_and_length \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15
```

Key reward environment variables:

| Variable | Default | Description |
|---|---|---|
| `MEILISEARCH_URL` | `http://localhost:7700` | MeiliSearch endpoint |
| `MEILI_ICD_INDEX` | `mimic_icd_10_first_diagnosis_codes` | Index name |
| `MATCH_REWARD` | `15.0` | Reward per exact ICD match |
| `NO_MATCH_MALUS` | `-0.2` | Penalty per unmatched prediction |
| `LENGTH_MALUS` | `-1e-4` | Per-character length penalty |
| `ACTIVATE_OUTCOME_REWARD` | `True` | Enable/disable ICD matching reward |
| `ACTIVATE_FORMAT_REWARD` | `True` | Enable/disable format reward |

### TRL

```bash
python src/train_clinical_r1_zero.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --train_data data/processed/train_prospective_icd_coding.parquet \
    --eval_data data/processed/val_prospective_icd_coding.parquet \
    --use_llm_as_a_judge false \
    --output_dir outputs/deep-icd-r1-7b
```

---

## Evaluation

```bash
python src/evaluate.py \
    --model_name <model_or_checkpoint_path> \
    --data data/processed/test_prospective_icd_coding.parquet \
    --output_dir outputs/eval/ \
    --batch_size 16 \
    --multiple_think_traces true
```

Results are logged to Weights & Biases and saved as a parquet file with per-example metrics.

---

## Tests

```bash
pytest test/
```

The test suite covers reward functions and partial overlap matching without requiring a live MeiliSearch instance (mocked).

---

## LLM-as-a-Judge (Optional)

Start a vLLM server with a Llama-3 model and set:

```bash
export VLLM_ENDPOINT=http://localhost:8000/v1
export JUDGE_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export JUDGE_PROMPT_FILE=src/prompts/llm_judge_rag_prompt.json
export JUDGE_RAG_ENABLED=True
export GUIDELINES_FILE=data/icd_10_cm_guidelines/  # path prefix
```

Then use `verl_batched_compute_score_multiple_think_traces_and_length_and_llm` as the reward function (verl) or set `use_llm_as_a_judge=true` (TRL).

---

## Citation

```bibtex
@article{deepicd-r1-2025,
  title   = {DeepICD-R1: Reinforcement Learning for Prospective ICD-10 Clinical Coding},
  author  = {DATEXIS},
  year    = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

Data use is subject to the [MIMIC-IV Data Use Agreement](https://physionet.org/content/mimiciv/view-required-training/1.0/).
