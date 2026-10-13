"""Guardrails: plain-Python checks, no extra models (6 GB GPU is already busy with vLLM).

Input side (before the model sees anything):
  - detect_injection: obvious prompt-injection / jailbreak phrases -> block
  - mask_pii: emails, phones, PAN, Aadhaar, card numbers -> placeholders like <EMAIL_1>
              The model and Langfuse only ever see the placeholders; real values are
              put back just before a tool runs.
Output side (after the model answers, before anything runs):
  - check_grounding: every argument value must come from the user's request
                     (catches made-up values like city="New York" for a joke)
  - risky tools (send_email) need explicit confirmation -> handled in graph.py
"""
import re

# ------------------------------------------------------------------ input

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (all |the )?(previous|prior|above|system)",
    r"(reveal|show|print|repeat) (me )?(your |the )?(system )?prompt",
    r"you are now\b", r"\bact as\b.*\b(dan|jailbreak|unfiltered)", r"developer mode",
    r"<\|im_(start|end)\|>",   # trying to inject Qwen chat-template tokens
]


def detect_injection(text: str) -> str | None:
    """Return the matched pattern if the text looks like an injection, else None."""
    low = text.lower()
    for pat in INJECTION_PATTERNS:
        if re.search(pat, low):
            return pat
    return None


# order matters: longer / more specific number patterns first
PII_PATTERNS = [
    ("EMAIL", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    ("PAN", r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ("CARD", r"\b(?:\d{4}[ -]?){3}\d{1,4}\b(?=\D|$)"),
    ("AADHAAR", r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    ("PHONE", r"(?:\+91[ -]?)?\b[6-9]\d{9}\b"),
]


def mask_pii(text: str) -> tuple[str, dict[str, str]]:
    """Replace PII with numbered placeholders. Returns (masked_text, {placeholder: real_value})."""
    mapping: dict[str, str] = {}
    for label, pat in PII_PATTERNS:
        def repl(m, label=label):
            key = f"<{label}_{sum(k.startswith(f'<{label}_') for k in mapping) + 1}>"
            mapping[key] = m.group(0)
            return key
        text = re.sub(pat, repl, text)
    return text, mapping


def unmask(value, mapping: dict[str, str]):
    """Put real values back into tool arguments (strings only)."""
    if isinstance(value, str):
        for key, real in mapping.items():
            value = value.replace(key, real)
    return value


# ------------------------------------------------------------------ output

# the model may legitimately normalise words into codes ("rupees" -> INR)
CURRENCY_ALIASES = {
    "INR": ["rupee", "rupees", "inr", "₹", "rs"], "USD": ["dollar", "usd", "$"],
    "EUR": ["euro", "eur", "€"], "GBP": ["pound", "gbp", "£"], "JPY": ["yen", "jpy", "¥"],
    "AED": ["dirham", "aed"], "CNY": ["yuan", "cny", "rmb"],
}
NUM = r"\d+(?:\.\d+)?"


def _numbers(text: str) -> set[float]:
    nums = {float(n) for n in re.findall(NUM, text.replace(",", ""))}
    # "15% of 2400" -> the model may write 0.15 * 2400 or 15/100 * 2400
    pcts = re.findall(rf"({NUM})\s*(?:%|percent)", text.replace(",", "").lower())
    if pcts:
        nums |= {float(n) / 100 for n in pcts} | {100.0}
    return nums


def _grounded(value, query: str) -> bool:
    q = query.lower()
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return float(value) in _numbers(query)
    if not isinstance(value, str):
        return True                                   # lists/dicts: not checked here
    v = value.lower().strip()
    if not v or v in q:
        return True
    if value.upper() in CURRENCY_ALIASES and any(a in q for a in CURRENCY_ALIASES[value.upper()]):
        return True
    nums_in_value = _numbers(value)
    if nums_in_value:                                 # e.g. "1250 * 18": every number must be in the query
        return nums_in_value <= _numbers(query)
    words = [w for w in re.findall(r"\w+", v) if len(w) > 2]
    return bool(words) and all(w in q for w in words)


# free-text args the model is expected to write itself (not copy from the query)
FREE_TEXT_ARGS = {("send_email", "subject"), ("send_email", "body")}


def check_grounding(calls: list[dict], query: str) -> list[str]:
    """Return a list like ['get_weather.city=New York'] for every argument not found in the query."""
    problems = []
    for call in calls:
        for arg, value in call.get("arguments", {}).items():
            if (call["name"], arg) in FREE_TEXT_ARGS:
                continue
            if not _grounded(value, query):
                problems.append(f"{call['name']}.{arg}={value}")
    return problems
