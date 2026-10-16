# server: tool-calling gateway (v1)

The production wrapper for the **SFT** model. It's a FastAPI + LangGraph gateway that turns a user message into a validated, guarded tool call, runs the tool, and replies. Every step is traced in Langfuse.

> For the DPO model with a general-chat fallback, use [`../serverv2`](../serverv2). This folder is kept as the v1 baseline.

## Flow
```
input_guard ─(injection)─► blocked
     │  mask PII → placeholders
call_model ─► validate ─(bad JSON / unknown tool / wrong args)─► refuse
                  │  └─(returned [])─► no_tool   canned "I can only help with…"
             output_guard ─(argument not in query)─► refuse
                  │      └─(risky tool, e.g. send_email)─► confirm
               execute ─► respond
```

## Files
| File | Role |
|---|---|
| `gateway.py` | FastAPI app: `POST /chat`, `POST /feedback`, `GET /health`, `GET /graph` |
| `graph.py` | LangGraph state machine (nodes above) |
| `guardrails.py` | Regex-based input/output checks: prompt injection, PII masking (email, phone, PAN, Aadhaar, card), argument grounding |
| `tools.py` | Real tools with free, keyless APIs: `get_weather` (Open-Meteo), `convert_currency` (Frankfurter), `calculate` (safe AST eval), `send_email` (mock) |
| `tracing.py` | Langfuse spans and scores; a no-op when no keys are set |
| `ui.py` | Gradio chat UI that talks to the gateway over HTTP |
| `smoke_test.py` | 5 queries against vLLM covering single, parallel and multi-tool calls |
| `gateway_v1.py` | First plain-FastAPI version (no graph), kept for reference |

## Run (from repo root)
```bash
# vLLM in WSL serving "sft-toolcaller" on :8000
uvicorn server.gateway:app --port 9000 --reload   # docs: http://localhost:9000/docs
GATEWAY_URL=http://localhost:9000 python -m server.ui
python -m server.smoke_test
```
Langfuse keys go in a root `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`). The `.env` file is git-ignored.

**Design note:** the guardrails are plain Python with no extra models, because the 6 GB GPU is already used by vLLM. For production upgrades, Presidio could handle PII and a Prompt-Guard-style classifier could handle injection.
