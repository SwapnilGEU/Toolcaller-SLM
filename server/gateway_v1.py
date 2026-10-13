"""Gateway v1: plain FastAPI, no LangGraph yet.

Flow for POST /chat:
  query -> build prompt (same as training) -> vLLM -> parse JSON -> validate
        -> run tools -> format a reply

Run from the repo root (serve env):
    uvicorn server.gateway:app --port 9000 --reload
Then open http://localhost:9000/docs
"""
import asyncio
import time
from typing import cast

from fastapi import FastAPI
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from src.data_utils import validate_tool_calls
from src.evaluate import extract_json
from src.prompts import build_messages
from server.tools import TOOL_FUNCS, TOOL_SCHEMAS

BACKENDS = {"vllm": ("http://localhost:8000/v1", "sft-toolcaller"),
            "ollama": ("http://localhost:11434/v1", "qwen2.5-0.5b-instruct-fc")}

app = FastAPI(title="SLM Tool-Calling Gateway")


class ChatRequest(BaseModel):
    query: str
    backend: str = "vllm"


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list | None      # what the model asked for
    tool_results: list           # what the tools returned
    issues: list[str]            # validation problems, empty = all good
    raw_model_output: str
    latency_ms: dict


def format_result(name: str, res: dict) -> str:
    """Templates instead of a second LLM: deterministic and free."""
    if "error" in res:
        return f"Sorry, {res['error']}."
    if name == "get_weather":
        return (f"{res['city']}, {res['country']}: {res['temperature_c']}°C, "
                f"{res['condition']}, wind {res['wind_kmh']} km/h.")
    if name == "convert_currency":
        return f"{res['amount']} {res['from']} = {res['result']:.2f} {res['to']}."
    if name == "calculate":
        return f"{res['expression']} = {res['result']}"
    return str(res)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t = {}
    base_url, model = BACKENDS[req.backend]
    client = AsyncOpenAI(base_url=base_url, api_key="none")

    # 1. model decides which tools to call
    t0 = time.perf_counter()
    messages = build_messages({"query": req.query, "tools": TOOL_SCHEMAS})
    resp = await client.chat.completions.create(
        model=model, messages=cast(list[ChatCompletionMessageParam], messages),
        temperature=0, max_tokens=256)
    raw = resp.choices[0].message.content or ""
    t["model"] = round((time.perf_counter() - t0) * 1000)

    # 2. parse + validate (never run anything the validator rejects)
    calls = extract_json(raw)
    if calls is None:
        return ChatResponse(reply="Sorry, I couldn't understand that request.", tool_calls=None,
                            tool_results=[], issues=["unparseable"], raw_model_output=raw, latency_ms=t)
    issues = validate_tool_calls(TOOL_SCHEMAS, calls)
    if issues:
        return ChatResponse(reply="Sorry, I couldn't build a valid action for that.", tool_calls=calls,
                            tool_results=[], issues=issues, raw_model_output=raw, latency_ms=t)

    # 3. no tool needed
    if not calls:
        return ChatResponse(reply="I can help with weather, currency conversion and calculations. "
                                  "Try asking about one of those!",
                            tool_calls=[], tool_results=[], issues=[], raw_model_output=raw, latency_ms=t)

    # 4. run each tool
    t0 = time.perf_counter()
    # run all calls at the same time instead of one after another
    outputs = await asyncio.gather(*(TOOL_FUNCS[c["name"]](**c["arguments"]) for c in calls))
    results = [{"name": c["name"], "result": r} for c, r in zip(calls, outputs)]
    t["tools"] = round((time.perf_counter() - t0) * 1000)

    reply = "\n".join(format_result(r["name"], r["result"]) for r in results)
    return ChatResponse(reply=reply, tool_calls=calls, tool_results=results, issues=[],
                        raw_model_output=raw, latency_ms=t)
