"""Gate logic of scripts/model_smoke_eval.py, tested with fake model answers (no torch, no download)."""
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "smoke", Path(__file__).resolve().parents[1] / "scripts" / "model_smoke_eval.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def frozen_samples(n=10):
    return (smoke.load_sample(smoke.EVAL_DIR / "test_v1.jsonl", n),
            smoke.load_sample(smoke.EVAL_DIR / "abstain_v1.jsonl", n))


def test_eval_sets_are_present_and_sampling_is_deterministic():
    test, abstain = frozen_samples()
    assert len(test) == len(abstain) == 10
    assert [r["id"] for r in test] == [r["id"] for r in frozen_samples()[0]]
    assert all(r["answers"] == [] for r in abstain)


def test_perfect_model_passes():
    test, abstain = frozen_samples()
    resp = {r["id"]: json.dumps(r["answers"]) for r in test + abstain}
    resp[smoke.JOKE["id"]] = "[]"
    report, passed = smoke.evaluate(resp, test, abstain, 65, 65)
    assert passed and report["exact_match"] == report["abstention"] == 100.0


def test_sft_like_model_that_never_abstains_fails():
    test, abstain = frozen_samples()
    resp = {r["id"]: json.dumps(r["answers"]) for r in test}
    resp |= {r["id"]: json.dumps([{"name": r["tools"][0]["name"], "arguments": {}}]) for r in abstain}
    resp[smoke.JOKE["id"]] = '[{"name": "get_weather", "arguments": {"city": "New York"}}]'
    report, passed = smoke.evaluate(resp, test, abstain, 65, 65)
    assert not passed and report["abstention"] == 0.0 and not report["joke_abstains"]
