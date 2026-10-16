"""The prompt is FROZEN: base, SFT and DPO were all evaluated with it.
If this test fails, every stored result is no longer comparable. Re-run the evals
(and update the hash) only if the change is intentional."""
import hashlib

from src.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_messages, format_answer

FROZEN_SHA256 = "975e22423c1cc5cb7c19eb970b74cb868e62e0a09ed42f1730464f5d6d478b0e"


def test_system_prompt_is_frozen():
    assert PROMPT_VERSION == "v1"
    assert hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest() == FROZEN_SHA256


def test_build_messages_shape():
    msgs = build_messages({"query": "weather in Pune?", "tools": [{"name": "get_weather"}]})
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "weather in Pune?" in msgs[1]["content"] and '"get_weather"' in msgs[1]["content"]


def test_format_answer_keeps_unicode():
    assert format_answer([]) == "[]"
    assert "₹" in format_answer([{"name": "x", "arguments": {"c": "₹"}}])
