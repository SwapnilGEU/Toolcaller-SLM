#!/usr/bin/env bash
# Fail if the repo tracks things that must never be pushed: weights, secrets, huge files.
set -euo pipefail
MAX_MB=${MAX_MB:-10}
fail=0

weights=$(git ls-files | grep -Ei '\.(safetensors|bin|pt|pth|ckpt|gguf|onnx)$' || true)
[ -n "$weights" ] && { echo "::error::Model weights are tracked:"; echo "$weights"; fail=1; }

envs=$(git ls-files | grep -E '(^|/)\.env(\..+)?$' | grep -v '\.example$' || true)
[ -n "$envs" ] && { echo "::error::.env files are tracked:"; echo "$envs"; fail=1; }

big=$(git ls-files -z | xargs -0 -r du -k 2>/dev/null | awk -v lim=$((MAX_MB*1024)) '$1>lim {printf "%6.1f MB  %s\n", $1/1024, $2}')
[ -n "$big" ] && { echo "::error::Files over ${MAX_MB} MB are tracked:"; echo "$big"; fail=1; }

# Hugging Face tokens, Langfuse secret keys, OpenAI-style keys
secrets=$(git grep -nIE '(hf_[A-Za-z0-9]{30,}|sk-lf-[A-Za-z0-9-]{20,}|sk-[A-Za-z0-9]{32,})' -- . ':!*.ipynb' || true)
nb_secrets=$(git grep -lIE '(hf_[A-Za-z0-9]{30,}|sk-lf-[A-Za-z0-9-]{20,})' -- '*.ipynb' || true)
[ -n "$secrets$nb_secrets" ] && { echo "::error::Possible secrets found in:"; echo "$secrets" | cut -c1-120; echo "$nb_secrets"; fail=1; }

[ $fail -eq 0 ] && echo "Repo hygiene OK: no weights, no .env, no secrets, nothing over ${MAX_MB} MB."
exit $fail
