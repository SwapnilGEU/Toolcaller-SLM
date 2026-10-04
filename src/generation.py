"""Fast, resumable batched generation for evaluation (and later DPO sampling).

Why each piece exists:
- batching: one question at a time leaves the GPU mostly idle (~9 tok/s on an RTX 4050 laptop);
  a batch of 16 processes 16 answers per step for roughly the same time per step.
- left padding: decoder-only models generate from the end of the prompt, so prompts of different
  lengths must be padded on the *left* so they all end at the same position.
- sort by prompt length: questions of similar length go in the same batch -> less padding wasted.
- JSON stop: each answer stops as soon as a complete JSON value has been written, instead of
  rambling to max_new_tokens (that rambling cost ~5 of the 6 hours in model selection).
- resumable: results are appended to disk after every batch; a crash or closed laptop resumes.
- OOM fallback: if a batch doesn't fit in 6 GB, the batch size is halved and retried.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


def json_is_complete(text):
    """True once the text contains a complete JSON array/object (starting at the first [ or {)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                decoder.raw_decode(text[i:])
                return True
            except json.JSONDecodeError:
                return False   # first bracket found but value not finished yet
    return False


class JsonCompleteCriteria(StoppingCriteria):
    """Per-sequence stop: each row in the batch finishes independently."""

    def __init__(self, tokenizer, prompt_len, check_every=1):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.check_every = check_every
        self.step = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.step += 1
        if self.step % self.check_every:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        return torch.tensor([json_is_complete(t) for t in texts], dtype=torch.bool, device=input_ids.device)


def prepare_tokenizer(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def generate_batch(model, tokenizer, message_lists, max_new_tokens, stop_at_json=True):
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in message_lists
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    if stop_at_json:
        kwargs["stopping_criteria"] = StoppingCriteriaList([JsonCompleteCriteria(tokenizer, prompt_len)])

    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, **kwargs)
    elapsed = time.perf_counter() - start

    generated = out[:, prompt_len:]
    responses = tokenizer.batch_decode(generated, skip_special_tokens=True)
    gen_tokens = (generated != tokenizer.pad_token_id).sum(dim=1).tolist()
    prompt_tokens = enc["attention_mask"].sum(dim=1).tolist()
    return responses, gen_tokens, prompt_tokens, elapsed


def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def run_generation(model, tokenizer, records, build_messages, out_path, run_config,
                   batch_size=16, max_new_tokens=512, stop_at_json=True, log_every=5):
    """Generate a response for every record; append rows to out_path (jsonl). Returns {id: row}.

    Rows already in out_path with the same run_config are reused, so re-running resumes.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sha = config_sha(run_config)

    done = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("run_config") == sha:
                    done[row["id"]] = row
        stale = sum(1 for _ in out_path.open(encoding="utf-8")) - len(done)
        if stale:
            print(f"Ignoring {stale} rows from a different config.")
            with out_path.open("w", encoding="utf-8") as f:   # keep only matching rows
                for row in done.values():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    todo = [r for r in records if r["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to generate (config {sha}).")
    if not todo:
        return done

    # similar lengths together -> less padding
    todo.sort(key=lambda r: len(json.dumps(build_messages(r), ensure_ascii=False)))

    prepare_tokenizer(tokenizer)
    start_all, i, n_batches = time.perf_counter(), 0, 0
    with out_path.open("a", encoding="utf-8") as f:
        while i < len(todo):
            batch = todo[i:i + batch_size]
            try:
                responses, gen_tokens, prompt_tokens, elapsed = generate_batch(
                    model, tokenizer, [build_messages(r) for r in batch], max_new_tokens, stop_at_json
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size == 1:
                    raise
                batch_size = max(1, batch_size // 2)
                print(f"Out of GPU memory -> batch size {batch_size}")
                continue

            for r, resp, g, p in zip(batch, responses, gen_tokens, prompt_tokens):
                row = {
                    "id": r["id"], "response": resp, "gen_tokens": int(g), "prompt_tokens": int(p),
                    "hit_max_tokens": int(g >= max_new_tokens),
                    "latency_per_example": elapsed / len(batch), "batch_size": len(batch),
                    "run_config": sha,
                }
                done[r["id"]] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            i += len(batch)
            n_batches += 1
            if n_batches % log_every == 0 or i >= len(todo):
                rate = i / (time.perf_counter() - start_all)
                eta_min = (len(todo) - i) / rate / 60 if rate else float("nan")
                print(f"  {i}/{len(todo)}  ({rate:.1f} examples/s, ~{eta_min:.0f} min left)")
    return done
