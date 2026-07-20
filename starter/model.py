"""
An optimized small GPT in plain PyTorch.
Complies with the strict 2,000,000 parameter constraint.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 1024      # byte-level tokenizer default (will update if custom vocab is built)
    block_size = 128
    n_layer = 5           # Increased from 4 to 5 using the tied parameter budget
    n_head = 4            # Evenly divides n_embd (160 / 5 = 32 per head dimension)
    n_embd = 160
    dropout = 0.05        # Introduced slight dropout for regularization
    tie_weights = True    # Tied input embedding and output projection layers

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Compute query, key, values for all heads in batch
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Fast causal scaled dot product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), 
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), 
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        
        # Enforce weight tying
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
            
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            # GPT-style initialization
            std = 0.02
            # Scaled initialization for residual projection layers
            if hasattr(m, 'weight') and m.weight.shape == (self.cfg.n_embd, self.cfg.n_embd):
                std *= 1 / math.sqrt(2 * self.cfg.n_layer)
                
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        
        # Compute embeddings
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        
        # Forward through transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            
        return logits, loss

    def n_params(self):
        # Dynamically calculates total parameters to monitor the hard cap
        return sum(p.numel() for p in self.parameters())