# src: shared library

Every notebook imports from here, so **base, SFT and DPO are prompted, generated and scored by identical code**. Without that, the comparisons wouldn't be valid.

| Module | What it does |
|---|---|
| `prompts.py` | The one system prompt and message builder (`PROMPT_VERSION = "v1"`) used for baseline, SFT data, DPO sampling and eval. Frozen after the baseline. |
| `data_utils.py` | JSON parsing, query normalization, tool-call schema validation (`validate_tool_calls`), call-pattern tagging, file hashing for split integrity |
| `abstention.py` | Builds "no tool fits → `[]`" examples. *Removed*: drop the gold tools and keep same-family distractors (hard). *Swapped*: pair the query with an unrelated tool list (easy). Skips records where a remaining tool could still answer. |
| `generation.py` | Fast batched eval generation: left padding, length-sorted batches, stop at the first complete JSON, resumable, halves the batch on OOM |
| `generate_cli.py` | Runs generation in a **fresh process**. Unsloth patches Qwen2 classes process-wide, which breaks plain-HF inference in the same kernel. Also merges LoRA into bf16. |
| `evaluate.py` | The single scorer: JSON extraction, exact / function / argument-F1 metrics, failure taxonomy, bootstrap CIs, McNemar test, per-feature breakdowns |

```python
from src.prompts import build_messages
from src.evaluate import score_prediction, summarize
```
