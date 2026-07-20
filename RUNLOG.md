# Training Run Log

## Run 1: Baseline Byte-Level Model
* **Hypothesis:** The starter GPT should train and evaluate end to end with the provided byte-level tokenizer and dev split.
* **What Changed:** Used the baseline starter configuration and trained a checkpoint for 2,000 steps under the project caps.
* **Dev bpb (Before/After):** N/A -> 2.32 bpb
* **Conclusion:** The baseline run proved the pipeline worked, but the score left room for improvement.

## Run 2: BPE Tokenization & Windows Optimization
* **Hypothesis:** A larger saved vocabulary/config and a slightly deeper GPT would improve the final dev score while staying under the 2M parameter cap.
* **What Changed:** The saved checkpoint now records `vocab_size=1024`, `block_size=128`, `n_layer=5`, `n_head=4`, `n_embd=160`, `dropout=0.05`, and `tie_weights=True`.
* **Dev bpb (Before/After):** 2.32 bpb -> 2.0934 bpb
* **Conclusion:** The final checkpoint evaluates cleanly with `python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt`, stays under the parameter cap at 1,731,040 parameters, and is the version to submit.