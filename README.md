# LLM 2,000 Step Speedrun 🚀

## Overview
This repository contains my submission for the LLM Speedrun challenge. The goal of this project was to train a custom GPT language model from scratch in PyTorch, strictly constrained to a maximum of **2,000,000 parameters** and **2,000 training steps**, while minimizing the Bits-Per-Byte (bpb) score on a mixed English/Hindi corpus.

## Core Architecture & Optimizations
* **Custom BPE Tokenizer:** Designed a pure-Python Byte-Pair Encoding (BPE) tokenizer with a 1,024 vocabulary size. I implemented $O(1)$ regex-based word-level caching to bypass $O(N^2)$ execution hangs, successfully compressing the 7.3M byte corpus by ~46%.
* **Architectural Restraint:** Built a ~1.1 million parameter GPT (5 layers, 4 attention heads, 128 embedding dimension, 128 context block size). By tying the embedding and LM head weights, the parameter budget was reallocated into deeper transformer layers while avoiding local CPU thread-thrashing.
* **Hardware-Specific Tuning:** Engineered specifically for stable execution on Windows local hardware. Bypassed PyTorch's `spawn` multiprocessing deadlocks by replacing `DataLoader` with manual tensor slicing, and removed native `bfloat16` emulation to run fast, unthrottled fp32 math.

## Deliverables & Repository Structure
* `train.py` & `model.py`: The core GPT architecture, AdamW optimizer, Cosine LR scheduler, and training loop.
* `tokenizer.py`: The custom BPE tokenizer implementation.
* `evaluate.py`: The evaluation script for calculating the final bpb metric.
* `ckpt.pt`: The final model checkpoint containing the 2,000-step training weights.
* `RUNLOG.md`: Detailed logs of the baseline vs. optimized training runs, including hypotheses and progression.
* `NOTES.md`: A concise summary of the winning configuration and architectural rationale.
* `SUMMARY.html`: A high-level overview of the optimization strategy and human-AI workflow.

## How to Run

Run these commands from the `starter/` directory.

**1. Train the model:**
\`\`\`bash
python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
\`\`\`

**2. Evaluate the model:**
\`\`\`bash
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
\`\`\`