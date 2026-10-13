"""Gateway v2: the tool-calling flow as a LangGraph graph.

                 ┌────────────┐
   START ──────► │ call_model │  your fine-tuned model writes JSON tool calls
                 └─────┬──────┘
                       ▼
                 ┌────────────┐
                 │  validate  │  parse JSON + check against tool schemas
                 └─────┬──────┘
          ┌────────────┼──────────────┐      (conditional edge: route())
          ▼            ▼              ▼
     ┌────────┐   ┌─────────┐   ┌─────────┐
     │ refuse │   │ no_tool │   │ execute │  run real Python tools
     └───┬────┘   └────┬────┘   └────┬────┘
         │             │             ▼
         │             │        ┌─────────┐
         │             │        │ respond │  template the results into a reply
         ▼             ▼        └────┬────┘
        END           END           END

Each node is a plain async function: it gets the full state and returns ONLY the
keys it changed. LangGraph merges that into the state and moves to the next node.
Guardrail nodes (step 4) will slot in before call_model and after validate.
"""
import asyncio
import operator
import time
from typing import Annotated, Any, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from server.tools import TOOL_FUNCS, TOOL_SCHEMAS
from src.data_utils import validate_tool_calls
from src.evaluate import extract_json
from src.prompts import build_messages

BACKENDS = {
    "vllm": ("http://localhost:8000/v1", "sft-toolcaller"),
    "ollama": ("http://localhost:11434/v1", "qwen2.5-0.5b-instruct-fc"),  # check `ollama list`
}
# one client per backend, created once (not per request) so connections get reused
CLIENTS = {name: AsyncOpenAI(base_url=url, api_key="none") for name, (url, _) in BACKENDS.items()}


def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict, total=False):
    # input
    query: str
    backend: str
    # filled in by nodes
    raw_model_output: str
    tool_calls: list | None
    issues: list[str]
    tool_results: list[dict]
    reply: str
    # Annotated[..., reducer]: instead of overwriting, LangGraph combines updates.
    # Every node adds its own timing and they all end up in one dict.
    latency_ms: Annotated[dict, merge_dicts]
    path: Annotated[list[str], operator.add]   # which nodes ran, for debugging/tracing


# ------------------------------------------------------------------ nodes

async def call_model(state: AgentState) -> dict:
    t0 = time.perf_counter()
    _, model = BACKENDS[state["backend"]]
    messages = build_messages({"query": state["query"], "tools": TOOL_SCHEMAS})
    resp = await CLIENTS[state["backend"]].chat.completions.create(
        model=model, messages=cast(list[ChatCompletionMessageParam], messages),
        temperature=0, max_tokens=256)
    return {"raw_model_output": resp.choices[0].message.content or "",
            "latency_ms": {"model": round((time.perf_counter() - t0) * 1000)},
            "path": ["call_model"]}


async def validate(state: AgentState) -> dict:
    calls = extract_json(state["raw_model_output"])
    if calls is None:
        issues = ["unparseable"]
    elif not isinstance(calls, list):
        issues = ["not_a_list"]
    else:
        issues = validate_tool_calls(TOOL_SCHEMAS, calls)
    return {"tool_calls": calls, "issues": issues, "path": ["validate"]}


def route(state: AgentState) -> str:
    """Conditional edge: the return value picks the next node."""
    if state["issues"]:
        return "refuse"
    if not state["tool_calls"]:
        return "no_tool"
    return "execute"


async def refuse(state: AgentState) -> dict:
    return {"reply": "Sorry, I couldn't build a valid action for that request.",
            "tool_results": [], "path": ["refuse"]}


async def no_tool(state: AgentState) -> dict:
    return {"reply": "I can help with weather, currency conversion and calculations. "
                     "Try asking about one of those!",
            "tool_results": [], "path": ["no_tool"]}


async def execute(state: AgentState) -> dict:
    t0 = time.perf_counter()
    calls = state["tool_calls"] or []

    async def run_one(call: dict[str, Any]) -> dict:
        try:
            return await TOOL_FUNCS[call["name"]](**call["arguments"])
        except Exception as e:          # one broken tool shouldn't crash the whole request
            return {"error": f"{call['name']} failed: {e}"}

    outputs = await asyncio.gather(*(run_one(c) for c in calls))
    return {"tool_results": [{"name": c["name"], "result": r} for c, r in zip(calls, outputs)],
            "latency_ms": {"tools": round((time.perf_counter() - t0) * 1000)},
            "path": ["execute"]}


def format_result(name: str, res: dict) -> str:
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


async def respond(state: AgentState) -> dict:
    reply = "\n".join(format_result(r["name"], r["result"]) for r in state["tool_results"])
    return {"reply": reply, "path": ["respond"]}


# ------------------------------------------------------------------ wiring

def build_graph():
    g = StateGraph(AgentState)
    for fn in (call_model, validate, refuse, no_tool, execute, respond):
        g.add_node(fn.__name__, fn)

    g.add_edge(START, "call_model")
    g.add_edge("call_model", "validate")
    g.add_conditional_edges("validate", route, ["refuse", "no_tool", "execute"])
    g.add_edge("execute", "respond")
    g.add_edge("refuse", END)
    g.add_edge("no_tool", END)
    g.add_edge("respond", END)
    return g.compile()


agent = build_graph()
