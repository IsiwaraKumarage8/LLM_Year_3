# Lab 03: Character-Level Tiny Transformer (`TinyStories`)

## Overview
This directory contains the implementation, training results, and serialized model checkpoint for **Lab 03 (Tiny Transformer)** in **PUSL 3205 — Large Language Models and Generative AI**. Built from scratch in PyTorch, this autoregressive character-level Transformer (`80,715` parameters) is trained on a 1,000-story subset of `roneneldan/TinyStories` to perform next-character prediction and coherent short-story generation.

---

## Repository Structure
```text
Lab-03-Tiny-Transformer/
├── README.md
├── notebooks/
│   └── tiny_transformer_tinystories.ipynb   # End-to-end PyTorch implementation & evaluation
└── results/
    ├── training_loss_curve.png              # Cross-entropy training vs. validation loss plot
    ├── generation_and_perplexity.png        # Perplexity evaluation & autoregressive text output
    └── tiny_transformer.pt                  # Serialized model state dictionary & vocabulary