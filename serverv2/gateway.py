"""Gateway v2: FastAPI is a thin HTTP layer. Logic -> serverv2/graph.py, guardrails ->
serverv2/guardrails.py, tracing -> serverv2/tracing.py. v2 = tool calling + general chat.

Run from the repo root (serve env), with the DPO model in vLLM (~/start_dpo_vllm.sh):
    uvicorn serverv2.gateway:app --port 8100 --reload
Optional env vars: VLLM_MODEL (default dpo-toolcaller), OLLAMA_MODEL, CHAT_ENABLED=0 (old canned reply).
Docs: http://localhost:9000/docs     Graph diagram: http://localhost:9000/graph
Langfuse keys go in a .env file in the repo root (see server/.env.example).
"""
try:
    from dotenv import load_dotenv
    load_dotenv()            # must run before serverv2.tracing creates the Langfuse client
except ImportError:
    pass

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel

from serverv2.graph import agent
from serverv2.tracing import ENABLED as TRACING, flush, run_traced, send_feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    flush()                  # send any buffered traces before shutting down


app = FastAPI(title="SLM Tool-Calling Gateway v2 (tools + chat)", version="4.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    query: str
    backend: str = "vllm"
    confirm: bool = False    # set true to approve risky tools (send_email)


class ChatResponse(BaseModel):
    reply: str
    mode: str                # tool | chat | no_tool | other (blocked / refused / needs confirmation)
    guardrail: str
    tool_calls: list | None
    tool_results: list
    issues: list[str]
    masked_query: str
    raw_model_output: str
    path: list[str]
    latency_ms: dict
    trace_id: str | None


class FeedbackRequest(BaseModel):
    trace_id: str
    good: bool
    comment: str | None = None


# Browsers always ask for "/" and "/favicon.ico" when you open the server's address.
# Those 404s were harmless noise; these two routes just silence them.
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)      # 204 = "no content", the browser stops asking


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
        mode=state.get("mode", "other"),
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


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if not TRACING:
        raise HTTPException(503, "Langfuse tracing is off (no LANGFUSE_PUBLIC_KEY), nowhere to store feedback")
    send_feedback(req.trace_id, req.good, req.comment)
    return {"ok": True}
