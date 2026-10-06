"""The single scorer for every stage (rule floor, base, SFT, DPO).

If base, SFT and DPO aren't scored by exactly the same code, their numbers aren't comparable.
So all scoring lives here and notebooks only import it.
"""
import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .data_utils import call_pattern, is_required, validate_tool_calls

# ------------------------------------------------------------------ parsing


def extract_json(text):
    """Dig the first complete JSON value out of a model response (handles markdown fences and chatter)."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except json.JSONDecodeError:
                continue
    return None


def normalize_calls(obj):
    """-> list of {"name", "arguments"} (possibly empty), or None if it isn't a valid tool-call structure."""
    if isinstance(obj, dict):
        obj = [obj] if "name" in obj else None
    if not isinstance(obj, list):
        return None

    calls = []
    for call in obj:
        if not isinstance(call, dict):
            return None
        name, arguments = call.get("name"), call.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return None
        calls.append({"name": name, "arguments": arguments})
    return calls


# ------------------------------------------------------------------ comparing calls


def _same_value(a, b, lenient):
    return str(a) == str(b) if lenient else a == b


def call_key(call, lenient=False):
    args = {k: str(v) for k, v in call["arguments"].items()} if lenient else call["arguments"]
    return json.dumps({"name": call["name"], "arguments": args}, sort_keys=True, ensure_ascii=False)


def calls_equal(pred, gold, lenient=False):
    """Order-insensitive: [A, B] == [B, A]. Compared as multisets."""
    return Counter(call_key(c, lenient) for c in pred) == Counter(call_key(c, lenient) for c in gold)


def argument_f1(pred, gold, lenient=False):
    """Partial credit for arguments.

    Gold and predicted calls *of the same name* are paired up, so order doesn't matter.
    Pairing is global: exact matches first, then the biggest overlaps. (Pairing each gold call
    greedily in order is wrong when one tool is called twice with overlapping arguments, e.g.
    v1_recipe(query) and v1_recipe(query, offset): it can pair them crosswise.)
    A hit = same argument name with the same value.
    precision = hits / predicted args, recall = hits / gold args.
    """
    def overlap(p, g):
        return sum(
            1 for k, v in g["arguments"].items()
            if k in p["arguments"] and _same_value(p["arguments"][k], v, lenient)
        )

    candidates = []
    for gi, g in enumerate(gold):
        for pj, p in enumerate(pred):
            if p["name"] == g["name"]:
                h = overlap(p, g)
                exact = h == len(g["arguments"]) == len(p["arguments"])
                candidates.append((exact, h, -abs(len(p["arguments"]) - len(g["arguments"])), gi, pj))
    candidates.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)

    used_g, used_p, hits = set(), set(), 0
    for exact, h, _, gi, pj in candidates:
        if gi not in used_g and pj not in used_p:
            used_g.add(gi)
            used_p.add(pj)
            hits += h

    n_pred = sum(len(p["arguments"]) for p in pred)
    n_gold = sum(len(g["arguments"]) for g in gold)
    if n_pred == 0 and n_gold == 0:   # e.g. tools without parameters, or [] vs []
        return float(Counter(p["name"] for p in pred) == Counter(g["name"] for g in gold))

    precision = hits / n_pred if n_pred else 0.0
    recall = hits / n_gold if n_gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


# ------------------------------------------------------------------ scoring one answer

FAILURE_ORDER = [
    "correct", "parse_fail", "called_tool_when_none_needed", "no_call_when_needed",
    "invented_tool", "wrong_call_count", "wrong_tool",
    "missing_required_arg", "unknown_arg", "wrong_value",
]


def classify_failure(pred, gold, tools, issues):
    """The *first* thing that went wrong, in FAILURE_ORDER."""
    if pred is None:
        return "parse_fail"
    if not gold and pred:
        return "called_tool_when_none_needed"
    if gold and not pred:
        return "no_call_when_needed"
    available = {t.get("name") for t in tools}
    if any(c["name"] not in available for c in pred):
        return "invented_tool"
    if len(pred) != len(gold):
        return "wrong_call_count"
    if Counter(c["name"] for c in pred) != Counter(c["name"] for c in gold):
        return "wrong_tool"
    if "missing_required" in issues:
        return "missing_required_arg"
    if "unknown_argument" in issues:
        return "unknown_arg"
    return "wrong_value"


def score_prediction(response_text, record):
    """Score one raw model response against one record ({"tools", "answers"})."""
    gold = normalize_calls(record["answers"])
    tools = record["tools"]
    pred = normalize_calls(extract_json(response_text))

    if pred is None:
        issues = ["unparsed"]
    else:
        issues = validate_tool_calls(tools, pred)

    tool_params = {t.get("name"): (t.get("parameters") or {}) for t in tools if isinstance(t, dict)}
    exact = pred is not None and calls_equal(pred, gold)

    return {
        "parse_valid": int(pred is not None),
        "schema_valid": int(pred is not None and not ({"unknown_tool", "unknown_argument", "missing_required"} & set(issues))),
        "function_match": int(pred is not None and Counter(c["name"] for c in pred) == Counter(c["name"] for c in gold)),
        "exact_match": int(exact),
        "exact_match_lenient": int(pred is not None and calls_equal(pred, gold, lenient=True)),
        "argument_f1": argument_f1(pred, gold) if pred is not None else 0.0,
        "invented_tool": int(pred is not None and any(c["name"] not in tool_params for c in pred)),
        "invented_argument": int(pred is not None and any(
            arg not in tool_params[c["name"]]
            for c in pred if c["name"] in tool_params
            for arg in c["arguments"]
        )),
        "n_pred_calls": len(pred) if pred is not None else None,
        "failure_type": "correct" if exact else classify_failure(pred, gold, tools, issues),
    }


# ------------------------------------------------------------------ dataset features (for breakdowns)


def query_template(query):
    """Mask numbers and quoted strings: 'Is 27 a power of 3?' -> 'is <N> a power of <N>?'."""
    q = " ".join(str(query).lower().split())
    q = re.sub(r"'[^']*'|\"[^\"]*\"", "<S>", q)
    return re.sub(r"\d+(\.\d+)?", "<N>", q)


def tag_features(record, train_templates=None):
    answers, tools = record["answers"], record["tools"]
    n_calls, n_tools = len(answers), len(tools)

    features = {
        "call_pattern": call_pattern(answers),
        "calls_bucket": str(n_calls) if n_calls < 4 else "4+",
        "tools_bucket": "1" if n_tools <= 1 else ("2-3" if n_tools <= 3 else "4+"),
        "arg_complexity": "structured" if any(
            isinstance(v, (list, dict)) for c in answers for v in (c.get("arguments") or {}).values()
        ) else "scalar",
        "has_optional": "yes" if any(
            not is_required(info) for t in tools for info in (t.get("parameters") or {}).values()
        ) else "no",
    }
    if train_templates is not None:
        features["template_twin"] = "seen_template" if query_template(record["query"]) in train_templates else "novel"
    return features


# ------------------------------------------------------------------ aggregating

HEADLINE = "exact_match"
METRICS = [
    "exact_match", "exact_match_lenient", "function_match", "argument_f1",
    "parse_valid", "schema_valid", "invented_tool", "invented_argument",
]


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """95% confidence interval of the mean by resampling the questions."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(scored_df, metrics=METRICS, n_boot=2000):
    """One row per metric: value and 95% CI, all in %."""
    rows = []
    for m in metrics:
        lo, hi = bootstrap_ci(scored_df[m], n_boot=n_boot)
        rows.append({"metric": m, "value": 100 * scored_df[m].mean(), "ci_low": 100 * lo, "ci_high": 100 * hi})
    return pd.DataFrame(rows).round(1).set_index("metric")


def breakdown(scored_df, feature, metric=HEADLINE):
    g = scored_df.groupby(feature)[metric]
    return pd.DataFrame({"n": g.size(), metric: (100 * g.mean()).round(1)}).sort_values("n", ascending=False)


def failure_mix(scored_df):
    mix = (100 * scored_df["failure_type"].value_counts(normalize=True)).round(1)
    return mix.reindex([f for f in FAILURE_ORDER if f in mix.index])


def _mcnemar_exact_p(b, c):
    """Two-sided exact McNemar test (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare_runs(df_a, df_b, metric=HEADLINE, key="id"):
    """Paired comparison on the same questions: what B fixed and broke relative to A."""
    merged = df_a[[key, metric]].merge(df_b[[key, metric]], on=key, suffixes=("_a", "_b"))
    a, b = merged[f"{metric}_a"].astype(bool), merged[f"{metric}_b"].astype(bool)
    fixed, broke = int((~a & b).sum()), int((a & ~b).sum())
    return {
        "n_paired": len(merged),
        "a": round(float(100 * a.mean()), 1),
        "b": round(float(100 * b.mean()), 1),
        "fixed": fixed,
        "broke": broke,
        "both_correct": int((a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),
        "mcnemar_p": round(_mcnemar_exact_p(fixed, broke), 4),
    }


GEN_FIELDS = ["response", "gen_tokens", "prompt_tokens", "hit_max_tokens", "latency_per_example"]


def score_generations(generations, records_by_id, features_df=None):
    """run_generation() output -> one scored row per question (+ features). Same table for base, SFT, DPO."""
    df = pd.DataFrame([
        {"id": rid, **{k: g[k] for k in GEN_FIELDS}, **score_prediction(g["response"], records_by_id[rid])}
        for rid, g in generations.items()
    ])
    return df.merge(features_df, on="id") if features_df is not None else df
