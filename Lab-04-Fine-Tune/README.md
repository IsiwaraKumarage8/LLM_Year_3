# Lab 04: Parameter-Efficient Fine-Tuning (LoRA) — `Qwen2.5-0.5B-Instruct`

## Overview
This directory contains the supervised fine-tuning (SFT) implementation, saved LoRA adapter weights, and evaluation results for **Lab 04 (Fine-Tuning an LLM using LoRA/QLoRA)** in **PUSL 3205 — Large Language Models and Generative AI**. Using Hugging Face's `peft` and `trl` (`SFTTrainer`) libraries, `Qwen/Qwen2.5-0.5B-Instruct` was fine-tuned on a custom 30-pair domain Q&A dataset (**Campus Essentials**, a university campus store) while updating only `0.8826%` of total model parameters.

---

## Repository Structure
```text
Lab-04-Fine-Tune/
├── README.md
├── notebooks/
│   └── lora_finetuning_qwen.ipynb          # End-to-end LoRA SFT pipeline & model evaluation
├── screenshots/
│   ├── 01_lora_trainable_params.png        # PEFT trainable parameter breakdown (0.8826%)
│   ├── 02_sft_training_loss.png            # SFTTrainer loss progression across 3 epochs
│   ├── 03_base_vs_finetuned_q1_q2.png      # Qualitative comparison on unseen domain queries (Q1–Q2)
│   └── 04_base_vs_finetuned_q3_q4.png      # Qualitative comparison on unseen domain queries (Q3–Q4)
└── uni_fundamentals_lora/
    └── final_adapter/                      # Serialized LoRA adapter weights & tokenizer files