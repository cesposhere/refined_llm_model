"""
Optimized trainer for the 2,000 Step LLM Speedrun.
Implements AdamW, Cosine LR Scheduling, and Gradient Clipping.
"""
import argparse
import time
import math

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

torch.set_num_threads(4)

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000

def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)  # Increased for better gradient estimates
    ap.add_argument("--lr", type=float, default=1e-3) # Increased peak LR for AdamW schedule
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()
    
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens (vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # 1. Advanced Optimizer Setup with Weight Decay Separation
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Embedding)
    blacklist_weight_modules = (torch.nn.LayerNorm,)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                if 'pos_emb' in fpn:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    
    # FIX: named_parameters() deduplicates tied weights automatically!
    # We must unconditionally discard the 'head.weight' string from our sets 
    # so the optimizer doesn't try to look for a key that no longer exists.
    decay.discard('head.weight')
    no_decay.discard('head.weight')
    param_dict.pop('head.weight', None)

    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    opt = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # 2. Cosine Learning Rate Schedule with Warmup
    warmup_steps = 100
    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / warmup_steps
        if step > args.steps:
            return args.lr * 0.1
        progress = (step - warmup_steps) / (args.steps - warmup_steps)
        return args.lr * 0.1 + args.lr * 0.45 * (1.0 + math.cos(math.pi * progress))

    model.train()
    t0 = time.time()
    losses = []
    
    print("Beginning optimized training loop...")
    for step in range(1, args.steps + 1):
        # Update learning rate per step
        lr = get_lr(step)
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        x, y = get_batch(ids, cfg.block_size, args.batch, device)
        
        # Removed autocast emulator to prevent CPU hanging
        _, loss = model(x, y)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        
        # 3. Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()
        losses.append(loss.item())
        
        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            # Added flush=True to force Windows to print immediately
            print(f"step {step:5d} | loss {avg:.4f} | lr {lr:.2e} | {(time.time()-t0)/step*1000:.0f} ms/step", flush=True)

    # Save Checkpoint
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()