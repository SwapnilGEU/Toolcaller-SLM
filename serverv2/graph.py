"""Gateway graph v2: tool calling + general chat (SFT + DPO model).

What changed vs server/graph.py: the DPO model answers [] when no tool fits (82% on the
frozen abstain set, notebook 10), so [] is now a reliable "this is not a tool request" signal.
Instead of a canned "I can only do weather/currency/..." reply, those requests go to
general_chat: the SAME model, called WITHOUT the tool system prompt, answers as a normal
assistant (0% tool-JSON leakage on 50 everyday prompts in notebook 10).
Set CHAT_ENABLED=0 to get the old canned reply back.

   START
     │
     ▼
 input_guard ──(injection)──► blocked ──► END        mask PII, block injections
     │
     ▼
 call_model                                          your fine-tuned model -> JSON
     │
     ▼
 validate ──(bad JSON / unknown tool / wrong args)──► refuse ──► END
     │  └──(returned [] = no tool fits)──► general_chat ──► END    same model, plain chat
     ▼
 output_guard ──(argument not in query)──► refuse ──► END
     │        └──(risky tool, not confirmed)──► confirm ──► END
     ▼
 execute ──► respond ──► END                         run real tools, template reply

Each node gets the full state and returns ONLY the keys it changed.
Each node is also a Langfuse span (@observe). capture_input=False everywhere because
the state holds the PII map, which must never be logged.
"""
import asyncio
import operator
import os
import time
from typing import Annotated, Any, TypedDict, cast

try:
    from langfuse.openai import AsyncOpenAI
except ImportError:
    from openai import AsyncOpenAI
from langgraph.graph import END, START, StateGraph
from openai.types.chat import ChatCompletionMessageParam

from serverv2.guardrails import check_grounding, detect_injection, mask_pii, unmask
from serverv2.tools import RISKY_TOOLS, TOOL_FUNCS, TOOL_SCHEMAS
from serverv2.tracing import ENABLED as TRACING, langfuse, observe
from src.data_utils import validate_tool_calls
from src.evaluate import extract_json, normalize_calls
from src.prompts import build_messages

# model names come from env vars, so switching models (sft-toolcaller / dpo-toolcaller) needs no code change
BACKENDS = {
    "vllm": (os.getenv("VLLM_URL", "http://localhost:8000/v1"), os.getenv("VLLM_MODEL", "dpo-toolcaller")),
    "ollama": (os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
               os.getenv("OLLAMA_MODEL", "qwen2.5_0.5b-instruct-fc-Chat:dpo")),
}
CHAT_ENABLED = os.getenv("CHAT_ENABLED", "1") != "0"
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.3"))   # a little variety; 0 tends to repeat itself
CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "300"))
CLIENTS = {name: AsyncOpenAI(base_url=url, api_key="none") for name, (url, _) in BACKENDS.items()}


def merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


class AgentState(TypedDict, total=False):
    # input
    query: str
    backend: str
    confirm: bool                    # user already approved risky actions
    # guardrails
    masked_query: str
    pii_map: dict[str, str]          # never logged
    guardrail: str                   # pass | blocked_injection | ungrounded | needs_confirmation
    mode: str                        # tool | chat | no_tool (which kind of answer the user got)
    # model + tools
    raw_model_output: str
    tool_calls: list | None
    issues: list[str]
    tool_results: list[dict]
    reply: str
    latency_ms: Annotated[dict, merge_dicts]
    path: Annotated[list[str], operator.add]


# ------------------------------------------------------------------ nodes

@observe(name="input_guard", as_type="guardrail", capture_input=False, capture_output=False)
async def input_guard(state: AgentState) -> dict:
    masked, pii_map = mask_pii(state["query"])
    hit = detect_injection(state["query"])
    if TRACING:
        langfuse.update_current_span(input=masked,
                                     output={"injection": hit, "pii_masked": sorted(pii_map)})
    return {"masked_query": masked, "pii_map": pii_map,
            "guardrail": "blocked_injection" if hit else "pass",
            "path": ["input_guard"]}


@observe(name="blocked", capture_input=False)
async def blocked(state: AgentState) -> dict:
    return {"reply": "I can't help with that request.", "tool_results": [], "path": ["blocked"]}


@observe(name="call_model", capture_input=False, capture_output=False)
async def call_model(state: AgentState) -> dict:
    t0 = time.perf_counter()
    _, model = BACKENDS[state["backend"]]
    # the model only ever sees the MASKED query
    messages = build_messages({"query": state["masked_query"], "tools": TOOL_SCHEMAS})
    resp = await CLIENTS[state["backend"]].chat.completions.create(
        name="toolcaller", model=model,
        messages=cast(list[ChatCompletionMessageParam], messages),
        temperature=0, max_tokens=256)
    return {"raw_model_output": resp.choices[0].message.content or "",
            "latency_ms": {"model": round((time.perf_counter() - t0) * 1000)},
            "path": ["call_model"]}


@observe(name="validate", capture_input=False)
async def validate(state: AgentState) -> dict:
    calls = extract_json(state["raw_model_output"])
    if calls is None:
        issues = ["unparseable"]
    elif not isinstance(calls, list):
        issues = ["not_a_list"]
    else:
        issues = validate_tool_calls(TOOL_SCHEMAS, calls)
    return {"tool_calls": calls, "issues": issues, "path": ["validate"]}


@observe(name="output_guard", as_type="guardrail", capture_input=False)
async def output_guard(state: AgentState) -> dict:
    calls = state["tool_calls"] or []
    ungrounded = check_grounding(calls, state["masked_query"])
    if ungrounded:
        return {"guardrail": "ungrounded", "issues": [f"ungrounded:{u}" for u in ungrounded],
                "path": ["output_guard"]}
    if any(c["name"] in RISKY_TOOLS for c in calls) and not state.get("confirm"):
        return {"guardrail": "needs_confirmation", "path": ["output_guard"]}
    return {"guardrail": "pass", "path": ["output_guard"]}


@observe(name="refuse", capture_input=False)
async def refuse(state: AgentState) -> dict:
    if state.get("guardrail") == "ungrounded":
        reply = ("I'm not confident about that: the action I came up with uses values "
                 "that aren't in your message. Could you rephrase with the details?")
    else:
        reply = "Sorry, I couldn't build a valid action for that request."
    return {"reply": reply, "tool_results": [], "path": ["refuse"]}


NO_TOOL_REPLY = ("I can help with weather, currency conversion, calculations and email. "
                 "Try asking about one of those!")


def looks_like_tool_json(text: str) -> bool:
    """True if a chat answer came back as tool-call JSON (or a bare []) instead of text."""
    return text.strip() == "[]" or bool(normalize_calls(extract_json(text)))


@observe(name="general_chat", capture_input=False, capture_output=False)
async def general_chat(state: AgentState) -> dict:
    """The tool-caller said [] (no tool fits) -> answer as a normal assistant.

    Same model, but WITHOUT the tool system prompt, so it answers in text (Qwen's default
    assistant prompt is used). It still only sees the MASKED query, and the input guard has
    already run, so injections never reach this node.
    """
    if not CHAT_ENABLED:
        return {"reply": NO_TOOL_REPLY, "mode": "no_tool", "tool_results": [], "path": ["no_tool"]}
    t0 = time.perf_counter()
    _, model = BACKENDS[state["backend"]]
    resp = await CLIENTS[state["backend"]].chat.completions.create(
        name="general_chat", model=model,
        messages=[{"role": "user", "content": state["masked_query"]}],
        temperature=CHAT_TEMPERATURE, max_tokens=CHAT_MAX_TOKENS)
    text = (resp.choices[0].message.content or "").strip()
    mode = "chat"
    if not text or looks_like_tool_json(text):        # safety net: never show raw JSON as a chat reply
        text, mode = NO_TOOL_REPLY, "no_tool"
    return {"reply": text, "mode": mode, "tool_results": [],
            "latency_ms": {"chat": round((time.perf_counter() - t0) * 1000)},
            "path": ["general_chat"]}


@observe(name="confirm", capture_input=False)
async def confirm(state: AgentState) -> dict:
    risky = [c for c in state["tool_calls"] if c["name"] in RISKY_TOOLS]
    lines = [f"- {c['name']}({', '.join(f'{k}={v!r}' for k, v in c['arguments'].items())})"
             for c in risky]
    return {"reply": "This needs your confirmation before I run it:\n" + "\n".join(lines)
                     + "\nSend the same request again with confirm=true to go ahead.",
            "tool_results": [], "path": ["confirm"]}


@observe(name="execute", as_type="tool", capture_input=False)
async def execute(state: AgentState) -> dict:
    t0 = time.perf_counter()
    calls = state["tool_calls"] or []
    pii = state.get("pii_map", {})

    async def run_one(call: dict[str, Any]) -> dict:
        args = {k: unmask(v, pii) for k, v in call["arguments"].items()}   # real values only here
        try:
            return await TOOL_FUNCS[call["name"]](**args)
        except Exception as e:
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
    if name == "send_email":
        return f"Email '{res['subject']}': {res['status']}."
    return str(res)


@observe(name="respond", capture_input=False)
async def respond(state: AgentState) -> dict:
    reply = "\n".join(format_result(r["name"], r["result"]) for r in state["tool_results"])
    return {"reply": reply, "mode": "tool", "path": ["respond"]}


# ------------------------------------------------------------------ routing

def after_input_guard(state: AgentState) -> str:
    return "blocked" if state["guardrail"] == "blocked_injection" else "call_model"


def after_validate(state: AgentState) -> str:
    if state["issues"]:
        return "refuse"
    if not state["tool_calls"]:
        return "general_chat"
    return "output_guard"


def after_output_guard(state: AgentState) -> str:
    return {"ungrounded": "refuse", "needs_confirmation": "confirm"}.get(state["guardrail"], "execute")


# ------------------------------------------------------------------ wiring

NODES = {"input_guard": input_guard, "blocked": blocked, "call_model": call_model,
         "validate": validate, "output_guard": output_guard, "refuse": refuse,
         "general_chat": general_chat, "confirm": confirm, "execute": execute, "respond": respond}


def build_graph(nodes=NODES):
    g = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", after_input_guard, ["blocked", "call_model"])
    g.add_edge("call_model", "validate")
    g.add_conditional_edges("validate", after_validate, ["refuse", "general_chat", "output_guard"])
    g.add_conditional_edges("output_guard", after_output_guard, ["refuse", "confirm", "execute"])
    g.add_edge("execute", "respond")
    for end_node in ("blocked", "refuse", "general_chat", "confirm", "respond"):
        g.add_edge(end_node, END)
    return g.compile()


agent = build_graph()
