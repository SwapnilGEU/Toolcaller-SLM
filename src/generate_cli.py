"""Run src.generation.run_generation in a FRESH Python process.

Why a separate process: when Unsloth loads a model for training, it replaces the forward
functions of the transformers model *classes* (Qwen2Attention etc.) for the whole process.
A plain transformers model loaded later in the same kernel runs Unsloth's patched code and
crashes (AttributeError: 'Qwen2Attention' object has no attribute 'apply_qkv'). A fresh process
never imports Unsloth, so it generates exactly like the baseline notebook did.

Merge only (base + LoRA adapter -> bf16 model on disk):
    python -m src.generate_cli --model Qwen/Qwen2.5-0.5B-Instruct --adapter ADAPTER --merged-out MERGED
Generate:
    python -m src.generate_cli --model MERGED --records in.jsonl --out preds.jsonl --run-config-file cfg.json
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_utils import read_jsonl
from .generation import run_generation
from .prompts import build_messages


def chat_messages(record):
    """Plain chat: no system prompt, no tools (general-behaviour check)."""
    return [{"role": "user", "content": record["query"]}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local folder")
    p.add_argument("--adapter", help="LoRA adapter folder to merge into --model")
    p.add_argument("--merged-out", help="where to save the merged model")
    p.add_argument("--records", help="jsonl of records to generate for (omit = merge only)")
    p.add_argument("--out", help="predictions jsonl (resumable)")
    p.add_argument("--run-config-file", help="json file with the run config")
    p.add_argument("--prompt", choices=["tools", "chat"], default="tools")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-json-stop", action="store_true")
    a = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto", device_map="auto")

    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
        if a.merged_out:
            model.save_pretrained(a.merged_out)
            tokenizer.save_pretrained(a.merged_out)
            print(f"Merged bf16 model saved -> {a.merged_out}")
    model.eval()

    if not a.records:
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_generation(
        model, tokenizer, read_jsonl(a.records),
        build_messages if a.prompt == "tools" else chat_messages,
        out_path=a.out, run_config=json.loads(Path(a.run_config_file).read_text()),
        batch_size=a.batch_size, max_new_tokens=a.max_new_tokens, stop_at_json=not a.no_json_stop,
    )
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps({"peak_vram_gb": peak}))
    if peak:
        print(f"Peak VRAM: {peak:.2f} GB")


if __name__ == "__main__":
    main()
