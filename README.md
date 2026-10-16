# Toolcaller-SLM

**A 0.5B model that calls tools accurately and knows when *not* to.**

Qwen2.5-0.5B-Instruct fine-tuned for function calling on [xLAM-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k): QLoRA SFT, then DPO to teach abstention. The model is served through vLLM or Ollama behind a LangGraph gateway that falls back to general chat.

| | |
|---|---|
| 🤗 Hugging Face | [SwapniL-pr1117/qwen2.5-0.5b-instructv2-fc](https://huggingface.co/SwapniL-pr1117/qwen2.5-0.5b-instructv2-fc) |
| 🦙 Ollama | `ollama run Avahan/qwen2.5_0.5b-instruct-fc-Chat` (tags `:sft`, `:dpo`, `:latest`) |

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

## Repo layout

```
notebooks/   numbered pipeline stages (1–6), each with its data/ and outputs/
src/         shared code: prompts, generation, evaluation, abstention scoring
server/      v1 gateway (tool calls only)
serverv2/    v2 gateway (tool calls + general-chat fallback)
assets/      README figures
```

Model weights, raw/processed data and checkpoints are git-ignored. Rebuild them with the notebooks or pull the models from HF/Ollama.

## Limitations

- About 18% of out-of-scope requests still trigger a tool call. Abstention is weaker when the right tool is *removed* (76.7%) than when it is *swapped* (88.0%).
- Rare repetition loops: parse failures are 0.3% (SFT: 0.1%).
- test_v1 mostly contains tool families seen in training, so generalization to unseen APIs isn't measured.
- vLLM eval numbers (`07_eval_vllm`) come from a separate stack and aren't mixed with the table above.

## Stack

Unsloth · TRL (SFT/DPO) · PEFT · Transformers · vLLM · llama.cpp/GGUF · Ollama · LangGraph · FastAPI · Langfuse
