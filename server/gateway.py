"""Gateway v2: FastAPI is now just a thin HTTP layer; the logic lives in server/graph.py.
(v1 with the logic inline is kept in gateway_v1.py for comparison.)

Run from the repo root (serve env):
    uvicorn server.gateway:app --port 9000 --reload
Docs: http://localhost:9000/docs     Graph diagram: http://localhost:9000/graph
"""
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from server.graph import agent

app = FastAPI(title="SLM Tool-Calling Gateway", version="2.0")


class ChatRequest(BaseModel):
    query: str
    backend: str = "vllm"


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list | None
    tool_results: list
    issues: list[str]
    raw_model_output: str
    path: list[str]          # nodes the request went through
    latency_ms: dict


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/graph", response_class=PlainTextResponse)
async def graph():
    """Mermaid source of the graph. Paste it into https://mermaid.live to see it."""
    return agent.get_graph().draw_mermaid()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    state = await agent.ainvoke({"query": req.query, "backend": req.backend,
                                 "latency_ms": {}, "path": []})
    return ChatResponse(
        reply=state["reply"],
        tool_calls=state.get("tool_calls"),
        tool_results=state.get("tool_results", []),
        issues=state.get("issues", []),
        raw_model_output=state.get("raw_model_output", ""),
        path=state["path"],
        latency_ms=state["latency_ms"],
    )
