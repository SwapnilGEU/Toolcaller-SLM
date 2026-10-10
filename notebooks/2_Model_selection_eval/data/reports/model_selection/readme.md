MODEL SELECTION RESULT
────────────────────────────────────────

Primary model:
Qwen2.5-0.5B-Instruct

Reason:
It achieved the strongest function-calling baseline by a large margin,
with 53.0% zero-shot exact match, 87.3% function accuracy, 95.6%
schema validity and 100% parse validity on the 181-example validation
benchmark.

Its performance remained substantially stronger across single-call,
parallel-call and multi-tool subsets.

SmolLM2-360M-Instruct is retained as a backup because it is small and
Apache-2.0 licensed, but its baseline function-calling accuracy is much
lower.

Gemma3-270M is dropped because of poor exact-match/function accuracy,
high parse/invented-tool failures and substantially higher latency.

LFM2-350M was not evaluated because the model could not be downloaded
during the benchmark run.