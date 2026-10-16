from src.data_utils import base_type, call_pattern, is_required, validate_tool_calls

TOOLS = [{"name": "f", "parameters": {
    "x": {"type": "int"}, "y": {"type": "str, optional"}, "z": {"type": "float", "default": 1.0}}}]


def test_is_required():
    p = TOOLS[0]["parameters"]
    assert is_required(p["x"]) and not is_required(p["y"]) and not is_required(p["z"])


def test_base_type():
    assert base_type("List[int]") == "list" and base_type("str, optional") == "str"
    assert base_type("weird") is None


def test_validate_tool_calls():
    assert validate_tool_calls(TOOLS, [{"name": "f", "arguments": {"x": 1}}]) == []
    assert validate_tool_calls(TOOLS, [{"name": "g", "arguments": {}}]) == ["unknown_tool"]
    assert validate_tool_calls(TOOLS, [{"name": "f", "arguments": {}}]) == ["missing_required"]
    assert validate_tool_calls(TOOLS, [{"name": "f", "arguments": {"x": 1, "w": 2}}]) == ["unknown_argument"]
    assert validate_tool_calls(TOOLS, [{"name": "f", "arguments": {"x": "1"}}]) == ["type_mismatch"]
    assert validate_tool_calls("nope", []) == ["invalid_tools"]


def test_call_pattern():
    a, b = {"name": "a"}, {"name": "b"}
    assert call_pattern([]) == "no_call"
    assert call_pattern([a]) == "single"
    assert call_pattern([a, a]) == "parallel_same_tool"
    assert call_pattern([a, b]) == "multi_distinct_tools"
    assert call_pattern([a, a, b]) == "mixed"
