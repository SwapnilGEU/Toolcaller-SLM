"""Shared helpers for the xLAM function-calling project.

Defined once here so the EDA/cleaning, model-selection and later eval notebooks
all judge tool calls with exactly the same rules.
"""
import json

# --------------------------------------------------------------------- JSON


def parse_json_field(value):
    """Parse a JSON string safely. Returns the parsed object, or None if invalid."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_query(query):
    """Lowercase + collapse whitespace, so trivial variants count as the same question."""
    if not isinstance(query, str):
        return ""
    return " ".join(query.lower().split())


# ------------------------------------------------------------- tool schemas

TYPE_CHECKS = {
    "str":   lambda v: isinstance(v, str),
    "int":   lambda v: isinstance(v, int) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool":  lambda v: isinstance(v, bool),
    "list":  lambda v: isinstance(v, list),
    "dict":  lambda v: isinstance(v, dict),
}


def base_type(type_str):
    """'str, optional' -> 'str', 'List[int]' / 'Tuple[..]' / 'set' -> 'list'. Unknown -> None (skip)."""
    if type_str is None:
        return None
    t = str(type_str).split(",")[0].strip().lower()
    if t.startswith(("list", "tuple", "set")):
        return "list"
    if t.startswith("dict"):
        return "dict"
    return t if t in TYPE_CHECKS else None


def is_required(param_info):
    """xLAM has no `required` key: required = not marked optional AND no non-empty default."""
    if not isinstance(param_info, dict):
        return False
    if "optional" in str(param_info.get("type", "")).lower():
        return False
    return param_info.get("default") in (None, "")


HARD_ISSUES = {
    "invalid_query", "invalid_tools", "empty_tools", "invalid_answers", "empty_answers",
    "tool_not_object", "tool_missing_name", "invalid_parameters", "conflicting_tool_definitions",
    "invalid_call", "unknown_tool", "invalid_arguments", "unknown_argument", "missing_required",
}
SOFT_ISSUES = {"type_mismatch"}


def validate_tool_calls(tools, answers):
    """Check every call in `answers` against the `tools` it was given. Returns a list of issue labels."""
    if not isinstance(tools, list):
        return ["invalid_tools"]
    if not isinstance(answers, list):
        return ["invalid_answers"]

    issues = []
    tool_map = {
        tool["name"]: tool.get("parameters") or {}
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }

    for call in answers:
        if not isinstance(call, dict):
            issues.append("invalid_call")
            continue
        if call.get("name") not in tool_map:
            issues.append("unknown_tool")
            continue

        params = tool_map[call["name"]]
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            issues.append("invalid_arguments")
            continue

        for arg_name, value in arguments.items():
            if arg_name not in params:
                issues.append("unknown_argument")
                continue
            param_info = params[arg_name]
            if not isinstance(param_info, dict):
                continue
            expected = base_type(param_info.get("type"))
            if expected and not TYPE_CHECKS[expected](value):
                issues.append("type_mismatch")

        for param_name, param_info in params.items():
            if is_required(param_info) and param_name not in arguments:
                issues.append("missing_required")

    return issues


# ------------------------------------------------------- dataset features


def call_pattern(answers):
    """single / parallel_same_tool / multi_distinct_tools / mixed."""
    if not isinstance(answers, list) or len(answers) == 0:
        return "no_call"
    if len(answers) == 1:
        return "single"
    names = [c.get("name") for c in answers if isinstance(c, dict)]
    if len(set(names)) == 1:
        return "parallel_same_tool"
    if len(set(names)) == len(names):
        return "multi_distinct_tools"
    return "mixed"
