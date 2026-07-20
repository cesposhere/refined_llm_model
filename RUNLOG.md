### Run 1: Architecture Optimization & Optimization Schedule
* **Hypothesis:** The baseline Adam optimizer (constant LR, no weight decay) and un-tied architecture waste the strict 2,000-step and 2M-parameter budget.
* **What Changed:** 
  1. Enabled weight tying between input embeddings and the output LM head.
  2. Reinvested saved parameters to increase network depth from 4 to 5 layers.
  3. Upgraded optimizer to AdamW with decoupled weight decay (0.1).
  4. Implemented a Cosine Annealing LR schedule with a 100-step linear warmup and gradient clipping (max_norm=1.0).
* **Conclusion:** The model trained stably without gradient explosions. The tied-weight architecture stayed well under the parameter cap (1.6M) while allowing a deeper network, bringing the training loss down smoothly to 1.72.

