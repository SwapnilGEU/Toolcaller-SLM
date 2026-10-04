"""The one prompt format used everywhere: baseline eval, SFT data formatting, DPO sampling, final eval.

Keeping it in a single module guarantees that "base vs SFT vs DPO" compares models,
not prompts. Change it here and every stage changes with it (and old results become
incomparable, so don't change it after the baseline has been run).
"""
import json

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a function-calling assistant.

Your task is to select the appropriate tool and provide the arguments required
to answer the user's request.

Return ONLY valid JSON.

The output must be a JSON array of tool calls in this format:

[
  {
    "name": "function_name",
    "arguments": {
      "argument_name": "value"
    }
  }
]

If none of the available tools can answer the request, return an empty array: []

Do not explain your answer.
Do not output markdown.
Do not invent tools.
"""


def build_user_prompt(record):
    tools_json = json.dumps(record["tools"], ensure_ascii=False, indent=2)
    return f"""Available tools:

{tools_json}

User request:

{record["query"]}

Return the required tool call(s) as JSON."""


def build_messages(record):
    """System + user turn. Used for generation (add_generation_prompt=True)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(record)},
    ]


def format_answer(answers):
    """The exact assistant output we train towards."""
    return json.dumps(answers, ensure_ascii=False)
