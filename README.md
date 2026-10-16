# Toolcaller-SLM

[![CI/CD](https://github.com/SwapnilGEU/Toolcaller-SLM/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/SwapnilGEU/Toolcaller-SLM/actions/workflows/ci_cd.yml)

**A 0.5B function-calling model that knows when *not* to call a tool.**

Small models fine-tuned for tool use learn one bad habit: *always call something*. Ask for a joke with only a weather API available, and you get the weather for New York. This project fine-tunes **Qwen2.5-0.5B-Instruct** on [xLAM-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) with **QLoRA SFT, then DPO**. The result is a model small enough for a laptop GPU that:

- gets tool calls exactly right **80%** of the time (base model: 61.6%)
- declines out-of-scope requests **82%** of the time (the SFT model: 0%)

It ships end to end. You can pull it from Hugging Face or Ollama, serve it with vLLM, and run it behind a guarded LangGraph gateway that hands non-tool requests to normal chat.

| | |
|---|---|
| 🤗 Hugging Face | [SwapniL-pr1117/qwen2.5-0.5b-instructv2-fc](https://huggingface.co/SwapniL-pr1117/qwen2.5-0.5b-instructv2-fc) |
| 🦙 Ollama | `ollama run Avahan/qwen2.5_0.5b-instruct-fc-Chat` (tags `:sft`, `:dpo`, `:latest`) |

![Gateway UI](assets/ui.png)

## Results

Greedy decoding, HF transformers bf16, same prompts for all three models.

| Metric | Base | SFT | **DPO** |
|---|---:|---:|---:|
| Exact match: test_v1 (n=2,929) | 61.6 | 78.0 | **80.0** |
| Function-name match | 89.2 | **97.3** | 96.8 |
| Argument F1 | 80.1 | 90.3 | **91.2** |
| Correct abstention: abstain_v1 (n=300) | 23.3 | 0.0 | **82.3** |
| Over-abstention on answerable calls | 0.2 | 0.0 | 0.2 |

- SFT boosts accuracy but **never abstains**: it calls *some* tool for every request.
- DPO restores abstention (0 → 82%) and still gains +2.0 exact match (95% CI 78.6–81.4; McNemar p = 0.0003 vs SFT).

![Base vs SFT vs DPO](assets/base_sft_dpo.png)

**Example:** "Tell me a joke", with only `get_weather` available:

| Model | Output |
|---|---|
| Base / SFT | `[{"name": "get_weather", "arguments": {"location": "New York"}}]` ❌ |
| DPO | `[]` ✅ → the gateway routes the request to general chat |

## Data & model selection

**Cleaning.** xLAM-60k looks clean but isn't:
- 812 examples have broken labels (missing required args, type mismatches, conflicting tool definitions)
- 367 are exact duplicates
- a naive random test split leaks 133 queries into training and caps achievable accuracy at 98.7%

The cleaning step drops the bad rows and builds a split with **0 leaked queries and 0 broken test labels**. A separate DPO pool is held out so preference data never touches SFT data.

![Data cleaning](assets/data_cleaning.png)

**Model selection.** Three sub-500M instruct models were tested zero-shot on 181 validation questions. Qwen2.5-0.5B was the only usable starting point. Few-shot prompting *hurt* every model (Qwen: 53 → 34% exact match), so all later work is zero-shot.

![Model selection](assets/model_selection.png)

| Model | Params | Exact match | Function match | Schema valid | p50 latency |
|---|---:|---:|---:|---:|---:|
| **Qwen2.5-0.5B** | 494M | **53.0** | **87.3** | **95.6** | **6.9 s** |
| SmolLM2-360M | 362M | 1.7 | 30.4 | 48.6 | 24.6 s |
| Gemma3-270M | 268M | 0.6 | 19.9 | 43.1 | 31.3 s |

## Pipeline

| Stage | Notebook | What it does |
|---|---|---|
| 1. Data | `1_Data_extraction/01_Data_cleaning` | Clean and validate xLAM-60k; EDA |
| 2. Model selection | `2_Model_selection_eval/02_model_selection`, `03_baseline_eval` | Compare Qwen2.5-0.5B, SmolLM2-360M and Gemma3-270M; baseline eval |
| 3. SFT data | `3_Data_Preperation_SFT/04_SFT_Data_Preperation_with_qwen_format` | Qwen chat format, leakage-free splits |
| 4. SFT | `4_Sft_Training/04_sft_training` | QLoRA (Unsloth) → merged SFT model |
| 5. DPO | `5_DPO_Training/06_dpo_data` → `08_dpo_training` → `09_eval_base_sft_dpo` | On-policy preference pairs, DPO training, 3-way eval |
| 6. Export | `6_Model_export/export_{huggingface,ollama}_{sft,dpo}` | HF Hub upload; GGUF q8_0 → Ollama |

**DPO details:**
- 1,753 preference pairs, of three kinds:
  - *mistake*: the SFT model's own wrong calls as rejected
  - *abstain*: requests no available tool fits, so `[]` is chosen
  - *balance*: answerable requests where a correct call is chosen, so the model doesn't learn to always say `[]`
- Training: new LoRA on the merged SFT model, with the adapter-disabled model as the reference. β = 0.1, RPO α = 1.0, lr 2e-5, 2 epochs.
- Held-out preference accuracy went from 0.49 to 0.78.

## Serving

```bash
# 1. Model server (vLLM, OpenAI-compatible)
~/start_dpo_vllm.sh                      # or: ollama run Avahan/qwen2.5_0.5b-instruct-fc-Chat:dpo

# 2. Gateway: tool routing, guardrails, general-chat fallback, Langfuse tracing
uvicorn serverv2.gateway:app --port 8100
python -m serverv2.ui                    # optional chat UI
```

`serverv2` sends each request to the model with the tool schema. If the model returns `[]`, the gateway answers it as normal chat. Every response carries a `mode` field (`tool` or `chat`). See [`serverv2/README.md`](serverv2/README.md).

**Feedback loop:** every answer in the UI has 👍 / 👎 buttons. A click is saved as a score on that request's Langfuse trace, next to the guardrail and schema-validity scores the gateway logs automatically. Filtering traces by 👎 or failed checks gives real-world mistakes. Paired with a corrected answer, they become new preference pairs for the next DPO round.

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci-cd.yml` | every push / PR | **Lint** (ruff) → **48 tests** on Python 3.10–3.12 → **repo hygiene** (no weights, `.env`, tokens or files >10 MB) → **Docker build + container health check** → on `main`/tags, **push the gateway image to GHCR** |
| `model-eval.yml` | manual / on release | Pulls the published model from HF and runs a **CPU smoke eval** with the same prompt, generation and scorer as the notebooks. Fails if exact match or abstention drops below 65% or "Tell me a joke" triggers a tool. |

What the tests cover:
- the scorer that produced every number above
- the frozen prompt (a hash check, so results stay comparable)
- abstention-data generation
- the guardrails (injection, PII, hallucinated arguments)
- the full LangGraph gateway with a fake model, so no GPU is needed

```bash
pip install -r requirements-dev.txt && pytest       # run locally
docker run -p 8100:8100 -e VLLM_URL=http://host.docker.internal:8000/v1 ghcr.io/swapnilgeu/toolcaller-slm-gateway
```

## Repo layout

| Folder | Contents |
|---|---|
| [`notebooks/`](notebooks) | Numbered pipeline stages 1–6, each with its own `data/` and `outputs/` |
| [`src/`](src) | Shared prompt, generation and scoring code, so every model is judged identically |
| [`server/`](server) | v1 gateway: tool calls only (SFT model) |
| [`serverv2/`](serverv2) | v2 gateway: tool calls + general-chat fallback (DPO model) |
| [`tests/`](tests) · `scripts/` | Test suite, model smoke eval, repo hygiene check |
| `assets/` | README figures |

Model weights, raw/processed data and checkpoints are git-ignored. Rebuild them with the notebooks or pull the models from HF/Ollama.

## Limitations

- About 18% of out-of-scope requests still trigger a tool call. Abstention is weaker when the right tool is *removed* (76.7%) than when it is *swapped* (88.0%).
- Rare repetition loops: parse failures are 0.3% (SFT: 0.1%).
- test_v1 mostly contains tool families seen in training, so generalization to unseen APIs isn't measured.
- vLLM eval numbers (`07_eval_vllm`) come from a separate stack and aren't mixed with the table above.

## Stack

Unsloth · TRL (SFT/DPO) · PEFT · Transformers · vLLM · llama.cpp/GGUF · Ollama · LangGraph · FastAPI · Langfuse
