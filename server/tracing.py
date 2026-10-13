"""Langfuse tracing. Kept separate so graph.py only needs one decorator per node.

How it fits together:
  - run_traced() opens ONE root observation per request (the "trace").
  - every graph node is decorated with @observe -> becomes a child span.
  - the model call uses langfuse.openai.AsyncOpenAI -> logged as a "generation"
    with prompt, output, token counts and latency, automatically.
  - at the end we attach scores to the trace (schema_valid, guardrail, ...).
    Later you filter traces by score in the Langfuse UI -> failures become DPO data.

If LANGFUSE_PUBLIC_KEY isn't set, the client is disabled and everything is a no-op,
so the gateway still works without Langfuse running.

Used LANGFUSE_HOST as cloud because of my RAM constraints.
"""
import os

from langfuse import get_client, observe  # noqa: F401  (observe re-exported for graph.py)

langfuse = get_client()
ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))


async def run_traced(agent, init_state: dict) -> tuple[dict, str | None]:
    with langfuse.start_as_current_observation(name="chat", as_type="agent") as root:
        state = await agent.ainvoke(init_state)

        # log the MASKED query only: real PII never reaches Langfuse
        root.update(input={"query": state.get("masked_query"), "backend": state["backend"]},
                    output={"reply": state.get("reply")},
                    metadata={"path": state["path"], "latency_ms": state["latency_ms"]})

        calls = state.get("tool_calls")
        issues = state.get("issues", [])
        root.score_trace(name="guardrail", value=state.get("guardrail", "pass"), data_type="CATEGORICAL")
        root.score_trace(name="parse_ok", value=0 if "unparseable" in issues else 1, data_type="BOOLEAN")
        root.score_trace(name="schema_valid", value=0 if issues else 1, data_type="BOOLEAN")
        root.score_trace(name="abstained", value=1 if calls == [] else 0, data_type="BOOLEAN")
        root.score_trace(name="n_tool_calls", value=len(calls or []), data_type="NUMERIC")

        trace_id = langfuse.get_current_trace_id() if ENABLED else None
    return state, trace_id


def send_feedback(trace_id: str, good: bool, comment: str | None = None):
    """Attach a human 👍/👎 to an existing trace. At DPO time: filter user_feedback = 0."""
    langfuse.create_score(trace_id=trace_id, name="user_feedback", value=1 if good else 0,
                          data_type="BOOLEAN", comment=comment)


def flush():
    langfuse.flush()
