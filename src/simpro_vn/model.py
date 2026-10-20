"""Shared recurrent-depth Transformer architecture."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x, offset=0):
        if offset + x.size(1) > self.pe.size(1):
            raise ValueError(f"Encoded input exceeds maximum length {self.pe.size(1)} (including BOS/EOS)")
        return x + self.pe[:, offset:offset + x.size(1)]


class CustomMultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x, mask=None, kv_cache=None):
        batch, length, _ = x.shape
        def project(layer):
            return layer(x).view(batch, length, self.nhead, self.d_k).transpose(1, 2)
        q, k, v = project(self.q_proj), project(self.k_proj), project(self.v_proj)
        if kv_cache is not None:
            k = torch.cat((kv_cache["k"], k), dim=2)
            v = torch.cat((kv_cache["v"], v), dim=2)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=(~mask if mask is not None else None),
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return self.out_proj(out), {"k": k, "v": v}


class CustomTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        attn_out, cache = self.self_attn(x, mask, kv_cache)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.dropout2(ff_out)), cache


class RecurrentValueNetwork(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, dim_feedforward=3072,
                 dropout=0.1, num_targets=17, max_seq_len=128):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.shared_transformer_block = CustomTransformerBlock(d_model, nhead, dim_feedforward, dropout)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_targets),
        )

    def forward(self, src, padding_mask=None, inference_depth=6, use_cache=False, past_caches=None):
        if inference_depth < 1:
            raise ValueError("inference_depth must be positive")
        if src.ndim != 2 or src.size(1) == 0:
            raise ValueError("src must have shape (batch, nonzero sequence length)")
        if past_caches is not None and not use_cache:
            raise ValueError("past_caches requires use_cache=True")
        if past_caches is not None and len(past_caches) != inference_depth:
            raise ValueError("Cache depth must equal inference_depth")
        offset = past_caches[0]["k"].size(2) if past_caches is not None else 0
        x = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model), offset)
        q_len, k_len = src.size(1), offset + src.size(1)
        # Offset-aware causal masking supports a single token AND a multi-token chunk.
        query_positions = torch.arange(offset, k_len, device=src.device)
        key_positions = torch.arange(k_len, device=src.device)
        mask = (key_positions[None, :] > query_positions[:, None])[None, None]
        if padding_mask is not None:
            if padding_mask.shape != (src.size(0), k_len):
                raise ValueError("padding_mask must cover all cached and current key positions")
            mask = mask | padding_mask[:, None, None, :].bool()
        caches = []
        for depth in range(inference_depth):
            previous = past_caches[depth] if past_caches is not None else None
            x, cache = self.shared_transformer_block(x, mask, previous)
            if use_cache:
                caches.append(cache)
        predictions = self.value_head(x)
        return (predictions, caches) if use_cache else predictions
