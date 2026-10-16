"""Smoke test for the vLLM server: 5 queries covering the main tool-calling patterns.

Run from the repo root (serve env, vLLM running in WSL):
    python -m serverv2.smoke_test
"""
import os
import time
from typing import cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.data_utils import validate_tool_calls
from src.evaluate import extract_json
from src.prompts import build_messages

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
MODEL = os.getenv("VLLM_MODEL", "dpo-toolcaller")   # set VLLM_MODEL=sft-toolcaller to test the SFT model

# Tools in the same shape as xLAM (a python list, NOT a json string:
# build_messages does the json.dumps itself).
TOOLS = [
    {"name": "get_weather", "description": "Get the current weather for a city.",
     "parameters": {"city": {"type": "str", "description": "City name"}}},
    {"name": "convert_currency", "description": "Convert an amount from one currency to another.",
     "parameters": {"amount": {"type": "float", "description": "Amount to convert"},
                    "from_currency": {"type": "str", "description": "3-letter source currency code, e.g. INR"},
                    "to_currency": {"type": "str", "description": "3-letter target currency code, e.g. USD"}}},
    {"name": "calculate", "description": "Evaluate a basic arithmetic expression.",
     "parameters": {"expression": {"type": "str", "description": "Expression like '23 * 47'"}}},
]

# (pattern, query, what we expect)
CASES = [
    ("single",        "What's the weather in Prayagraj?",                      "get_weather(Prayagraj)"),
    ("args",          "Convert 500 rupees to US dollars",                      "convert_currency(500, INR, USD)"),
    ("parallel_same", "How's the weather in Delhi and Mumbai right now?",      "get_weather x2"),
    ("multi_distinct","Weather in Goa, and what is 1250 * 18?",                "get_weather + calculate"),
    ("abstain",       "Tell me a joke about cats",                             "[] (no tool fits)"),
]

for i, (pattern, query, expected) in enumerate(CASES, 1):
    messages = build_messages({"query": query, "tools": TOOLS})
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=cast(list[ChatCompletionMessageParam], messages),
        temperature=0,
        max_tokens=256,
    )
    ms = (time.perf_counter() - t0) * 1000

    raw = resp.choices[0].message.content
    parsed = extract_json(raw)
    issues = validate_tool_calls(TOOLS, parsed) if parsed is not None else ["unparseable"]
    usage = resp.usage

    print(f"\n[{i}] {pattern}: {query}")
    print(f"    expected : {expected}")
    print(f"    raw      : {raw}")
    print(f"    valid    : {'yes' if not issues else issues}")
    token_counts = (
        f"{usage.prompt_tokens}/{usage.completion_tokens}" if usage is not None else "n/a"
    )
    print(f"    latency  : {ms:.0f} ms  |  tokens in/out: {token_counts}")
