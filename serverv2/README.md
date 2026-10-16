# serverv2: tool calling + general chat

Same gateway as `server/` (LangGraph + guardrails + Langfuse + Gradio), for the **SFT + DPO** model.
The only behaviour change: requests where no tool fits now get a **normal chat answer** instead of
"I can only help with weather, currency...".

## Why this works now
- The SFT model never returned `[]`: "Tell me a joke" became `get_weather(New York)`.
- The DPO model returns `[]` when no tool fits (82% on 300 frozen questions, `09_eval_base_sft_dpo`),
  so `[]` is a reliable signal for "this is chat, not a tool request".
- In plain chat (no tool system prompt) it answers in text: 0% tool-JSON leakage on 50 prompts.

## Flow
```
input_guard ─(injection)─► blocked
     │
call_model (tool prompt) ─► validate ─(bad JSON / wrong args)─► refuse
                                 │  └─(returned [])─► general_chat   same model, NO tool prompt
                                 ▼
                          output_guard ─(ungrounded)─► refuse
                                 │      └─(risky tool)─► confirm
                                 ▼
                           execute ─► respond
```
- `general_chat` only sees the **masked** query (PII placeholders), after the injection check.
- Safety net: if a chat answer still comes back as tool JSON, the old canned reply is shown instead.
- Every response has a `mode`: `tool` | `chat` | `no_tool` | `other`, also logged to Langfuse as a score.

## Run (repo root, `serve` env)
```bash
# WSL: ~/start_dpo_vllm.sh            (serves dpo-toolcaller on :8000)
uvicorn serverv2.gateway:app --port 8100 --reload
python -m serverv2.ui                 # http://localhost:7860
python -m serverv2.smoke_test         # quick check against vLLM
```

## Settings (env vars, optional)
| Variable | Default | Meaning |
|---|---|---|
| `VLLM_MODEL` | `dpo-toolcaller` | served model name (`sft-toolcaller` for the old model) |
| `OLLAMA_MODEL` | `qwen2.5_0.5b-instruct-fc-Chat:dpo` | Ollama backend model |
| `CHAT_ENABLED` | `1` | `0` = old canned "no tool" reply instead of chat |
| `CHAT_TEMPERATURE` | `0.3` | chat only; tool calls stay at temperature 0 |
| `CHAT_MAX_TOKENS` | `300` | chat answer length cap |

`server/` is unchanged and still works as before.
