# notebooks: the pipeline

Run the stages in order. Each stage folder keeps its own `data/` (reports and eval results are tracked; raw data and splits are git-ignored) and `outputs/` (weights, always git-ignored).

| # | Notebook | Output |
|---|---|---|
| **1** | `1_Data_extraction/01_Data_cleaning` | xLAM-60k → 58,620 clean examples (97.7% kept); leakage-free splits; `data/reports/cleaning_report.json` |
| **2** | `2_Model_selection_eval/02_model_selection` | Qwen2.5-0.5B vs SmolLM2-360M vs Gemma3-270M → Qwen picked (53% vs 1.7% vs 0.6% zero-shot exact match) |
| | `2_Model_selection_eval/03_baseline_eval` | Base-model scores on the frozen test set (n=2,929) |
| **3** | `3_Data_Preperation_SFT/04_SFT_Data_Preperation_with_qwen_format` | Chat-formatted SFT data plus abstention examples |
| **4** | `4_Sft_Training/04_sft_training` | QLoRA SFT (Unsloth) → merged SFT model + eval |
| | `4_Sft_Training/05_sft_training_optimized` | Optimized config, designed but not run |
| **5** | `5_DPO_Training/06_dpo_data` | 1,753 on-policy preference pairs (mistake / abstain / balance) |
| | `5_DPO_Training/07_eval_vllm` | SFT eval through vLLM (separate stack, not mixed with HF numbers) |
| | `5_DPO_Training/08_dpo_training` | DPO (β 0.1, RPO α 1.0) → merged DPO model |
| | `5_DPO_Training/09_eval_base_sft_dpo` | **Headline result:** base vs SFT vs DPO, with CIs and McNemar test |
| **6** | `6_Model_export/export_huggingface_{sft,dpo}` | Upload to the HF Hub with a model card |
| | `6_Model_export/export_ollama_{sft,dpo}` | GGUF q8_0 → Ollama (`:sft`, `:dpo`, `:latest`), rerunnable with hard timeouts |

**Environment:** Windows + RTX 4050 laptop GPU (6 GB). Training uses Unsloth/TRL; eval uses plain HF transformers in a fresh process (see `src/generate_cli.py`). Prototype and backup notebooks are kept locally and git-ignored.
