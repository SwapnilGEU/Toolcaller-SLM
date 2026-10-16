"""CPU smoke eval of the PUBLISHED model: a release gate, not a replacement for notebook 09.

Pulls the model from the Hugging Face Hub and scores a fixed random sample of the frozen eval sets
with the SAME prompt (src.prompts), generation (src.generation) and scorer (src.evaluate) as the
notebooks. Fails (exit 1) if tool-call accuracy or abstention drops below a threshold, or if the
model calls a tool for "Tell me a joke".

    python scripts/model_smoke_eval.py --model SwapniL-pr1117/qwen2.5-0.5b-instructv2-fc --n 40

Full-set reference (notebook 09, bf16 GPU): exact match 80.0, abstention 82.3. Thresholds sit
well below that because a 40-question sample has roughly ±12-point noise.
"""
import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import score_prediction  # noqa: E402

EVAL_DIR = ROOT / "notebooks" / "5_DPO_Training" / "data" / "eval"
JOKE = {
    "id": "showcase_joke", "query": "Tell me a joke", "answers": [],
    "tools": [{"name": "get_weather", "description": "Get the current weather for a city.",
               "parameters": {"city": {"type": "str", "description": "City name"}}}],
}


def load_sample(path, n, seed=0):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return random.Random(seed).sample(rows, min(n, len(rows)))


def evaluate(responses, test, abstain, min_exact, min_abstain):
    """Pure scoring + gating (unit-tested). responses: {id: model text}. Returns (report, passed)."""
    def pct(rows):
        return round(100 * sum(score_prediction(responses[r["id"]], r)["exact_match"] for r in rows) / len(rows), 1)

    report = {
        "exact_match": pct(test),
        "abstention": pct(abstain),
        "joke_abstains": score_prediction(responses[JOKE["id"]], JOKE)["exact_match"] == 1,
        "joke_response": responses[JOKE["id"]].strip()[:200],
        "n_test": len(test), "n_abstain": len(abstain),
    }
    checks = {
        f"exact match ≥ {min_exact}": report["exact_match"] >= min_exact,
        f"abstention ≥ {min_abstain}": report["abstention"] >= min_abstain,
        "'Tell me a joke' → []": report["joke_abstains"],
    }
    report["checks"] = checks
    return report, all(checks.values())


def generate(model_id, records, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.generation import run_generation
    from src.prompts import build_messages

    torch.set_num_threads(os.cpu_count() or 2)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    with tempfile.TemporaryDirectory() as tmp:
        rows = run_generation(model, tok, records, build_messages, Path(tmp) / "preds.jsonl",
                              run_config={"model": model_id, "ci": True},
                              batch_size=batch_size, max_new_tokens=256)
    return {rid: row["response"] for rid, row in rows.items()}


def write_summary(report, passed, model_id, seconds):
    lines = [
        f"## Model smoke eval: {'✅ passed' if passed else '❌ failed'}",
        f"`{model_id}` on CPU, {report['n_test']} test + {report['n_abstain']} abstain questions, {seconds:.0f}s",
        "", "| Check | Result |", "|---|---|",
        f"| Exact match (test_v1 sample) | {report['exact_match']}% |",
        f"| Abstention (abstain_v1 sample) | {report['abstention']}% |",
        f"| 'Tell me a joke' with only get_weather | `{report['joke_response']}` |",
        "", *[f"- {'✅' if ok else '❌'} {name}" for name, ok in report["checks"].items()],
    ]
    text = "\n".join(lines)
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SwapniL-pr1117/qwen2.5-0.5b-instructv2-fc")
    ap.add_argument("--n", type=int, default=40, help="questions per eval set")
    ap.add_argument("--min-exact", type=float, default=65.0)
    ap.add_argument("--min-abstain", type=float, default=65.0)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    test = load_sample(EVAL_DIR / "test_v1.jsonl", args.n)
    abstain = load_sample(EVAL_DIR / "abstain_v1.jsonl", args.n)
    t0 = time.time()
    responses = generate(args.model, test + abstain + [JOKE], args.batch_size)
    report, passed = evaluate(responses, test, abstain, args.min_exact, args.min_abstain)
    write_summary(report, passed, args.model, time.time() - t0)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
