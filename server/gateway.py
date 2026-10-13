"""Gateway: FastAPI is a thin HTTP layer. Logic -> server/graph.py, guardrails ->
server/guardrails.py, tracing -> server/tracing.py. (v1 kept in gateway_v1.py.)

Run from the repo root (serve env):
    uvicorn server.gateway:app --port 9000 --reload
Docs: http://localhost:9000/docs     Graph diagram: http://localhost:9000/graph
Langfuse keys go in a .env file in the repo root (see server/.env.example).
"""
try:
    from dotenv import load_dotenv
    load_dotenv()            # must run before server.tracing creates the Langfuse client
except ImportError:
    pass

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from server.graph import agent
from server.tracing import ENABLED as TRACING, flush, run_traced


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    flush()                  # send any buffered traces before shutting down


app = FastAPI(title="SLM Tool-Calling Gateway", version="3.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    backend: str = "vllm"
    confirm: bool = False    # set true to approve risky tools (send_email)


class ChatResponse(BaseModel):
    reply: str
    guardrail: str
    tool_calls: list | None
    tool_results: list
    issues: list[str]
    masked_query: str
    raw_model_output: str
    path: list[str]
    latency_ms: dict
    trace_id: str | None


@app.get("/health")
async def health():
    return {"status": "ok", "tracing": TRACING}


@app.get("/graph", response_class=PlainTextResponse)
async def graph():
    """Mermaid source of the graph. Paste into https://mermaid.live to see it."""
    return agent.get_graph().draw_mermaid()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    state, trace_id = await run_traced(agent, {
        "query": req.query, "backend": req.backend, "confirm": req.confirm,
        "latency_ms": {}, "path": []})
    return ChatResponse(
        reply=state["reply"],
        guardrail=state.get("guardrail", "pass"),
        tool_calls=state.get("tool_calls"),
        tool_results=state.get("tool_results", []),
        issues=state.get("issues", []),
        masked_query=state.get("masked_query", ""),
        raw_model_output=state.get("raw_model_output", ""),
        path=state["path"],
        latency_ms=state["latency_ms"],
        trace_id=trace_id,
    )
