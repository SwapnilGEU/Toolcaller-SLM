"""The scorer is shared by base / SFT / DPO: if it breaks, every number in the README is wrong."""
import pytest

from src.evaluate import (_mcnemar_exact_p, argument_f1, calls_equal, extract_json,
                          normalize_calls, score_prediction)

TOOLS = [{"name": "get_weather", "parameters": {"city": {"type": "str"}}},
         {"name": "add", "parameters": {"a": {"type": "int"}, "b": {"type": "int"}}}]


@pytest.mark.parametrize("text, expected", [
    ('[{"name": "add", "arguments": {"a": 1, "b": 2}}]', [{"name": "add", "arguments": {"a": 1, "b": 2}}]),
    ('```json\n[]\n```', []),
    ('Sure! Here you go: [{"name": "x", "arguments": {}}] hope it helps', [{"name": "x", "arguments": {}}]),
    ("no json here", None),
    (None, None),
])
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_normalize_calls_rejects_bad_structure():
    assert normalize_calls({"name": "a", "arguments": {}}) == [{"name": "a", "arguments": {}}]
    assert normalize_calls([{"name": 3}]) is None
    assert normalize_calls("text") is None
    assert normalize_calls([]) == []


def test_calls_equal_is_order_insensitive_and_lenient_on_types():
    a = [{"name": "add", "arguments": {"a": 1, "b": 2}}, {"name": "get_weather", "arguments": {"city": "Pune"}}]
    assert calls_equal(a, a[::-1])
    assert not calls_equal(a, a[:1])
    assert calls_equal([{"name": "add", "arguments": {"a": "1"}}], [{"name": "add", "arguments": {"a": 1}}], lenient=True)
    assert not calls_equal([{"name": "add", "arguments": {"a": "1"}}], [{"name": "add", "arguments": {"a": 1}}])


def test_argument_f1_partial_credit_and_crosswise_pairing():
    gold = [{"name": "add", "arguments": {"a": 1, "b": 2}}]
    assert argument_f1(gold, gold) == 1.0
    assert argument_f1([{"name": "add", "arguments": {"a": 1, "b": 9}}], gold) == pytest.approx(0.5)
    assert argument_f1([], []) == 1.0
    # same tool twice: must pair globally, not greedily in order
    g2 = [{"name": "r", "arguments": {"q": "x"}}, {"name": "r", "arguments": {"q": "x", "o": 1}}]
    assert argument_f1(g2[::-1], g2) == 1.0


def test_score_prediction_correct_call():
    rec = {"tools": TOOLS, "answers": [{"name": "add", "arguments": {"a": 1, "b": 2}}]}
    s = score_prediction('[{"name": "add", "arguments": {"b": 2, "a": 1}}]', rec)
    assert s["exact_match"] == s["function_match"] == s["schema_valid"] == 1
    assert s["failure_type"] == "correct"


@pytest.mark.parametrize("response, failure", [
    ("garbage", "parse_fail"),
    ("[]", "no_call_when_needed"),
    ('[{"name": "multiply", "arguments": {}}]', "invented_tool"),
    ('[{"name": "get_weather", "arguments": {"city": "Pune"}}]', "wrong_tool"),
    ('[{"name": "add", "arguments": {"a": 1}}]', "missing_required_arg"),
    ('[{"name": "add", "arguments": {"a": 1, "b": 5}}]', "wrong_value"),
])
def test_failure_taxonomy(response, failure):
    rec = {"tools": TOOLS, "answers": [{"name": "add", "arguments": {"a": 1, "b": 2}}]}
    assert score_prediction(response, rec)["failure_type"] == failure


def test_abstention_is_scored_like_any_other_answer():
    rec = {"tools": TOOLS, "answers": []}
    assert score_prediction("[]", rec)["exact_match"] == 1
    joke = score_prediction('[{"name": "get_weather", "arguments": {"city": "New York"}}]', rec)
    assert joke["exact_match"] == 0 and joke["failure_type"] == "called_tool_when_none_needed"


def test_mcnemar():
    assert _mcnemar_exact_p(0, 0) == 1.0
    assert _mcnemar_exact_p(163, 103) < 0.001     # the DPO vs SFT result in the README
