from src.abstention import build_abstention_records, make_removed, make_swapped, similar_names


def rec(i, tools, used):
    return {"id": i, "query": f"q{i}", "tools": [{"name": n, "description": d} for n, d in tools],
            "answers": [{"name": u, "arguments": {}} for u in used]}


def test_removed_drops_gold_tools_and_expects_empty_answer():
    r = rec(1, [("get_stock_price", "stock quote"), ("translate_text", "translate a sentence")], ["get_stock_price"])
    neg = make_removed(r)
    assert neg["answers"] == [] and [t["name"] for t in neg["tools"]] == ["translate_text"]


def test_removed_skips_when_a_lookalike_tool_could_still_answer():
    r = rec(1, [("auto_complete", "suggest words"), ("auto_complete_deprecating", "suggest words")], ["auto_complete"])
    assert similar_names("auto_complete", "auto_complete_deprecating")
    assert make_removed(r) is None


def test_swapped_uses_donor_tools():
    r = rec(1, [("get_stock_price", "stock quote")], ["get_stock_price"])
    donor = rec(2, [("translate_text", "translate a sentence")], ["translate_text"])
    assert make_swapped(r, donor)["tools"] == donor["tools"]


def test_build_is_deterministic():
    names = ["stock", "weather", "recipe", "movie", "flight", "translate", "lyrics", "crypto"]
    recs = [rec(i, [(f"get_{a}", f"{a} lookup"), (f"find_{b}", f"{b} search")], [f"get_{a}"])
            for i, (a, b) in enumerate(zip(names, names[1:] + names[:1]))]
    a = build_abstention_records(recs, 4, seed=1, id_prefix="neg_")
    assert a == build_abstention_records(recs, 4, seed=1, id_prefix="neg_")
    assert all(r["answers"] == [] for r in a)
