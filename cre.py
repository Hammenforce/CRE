"""
Complex Resonance Embedding (CRE) - PyTorch Implementation

A linear-complexity sequence processing architecture using push-pull dynamics
and multi-frequency resonance for efficient long-range dependency modeling.

Author: Sten Daniel Hammenfors
License: Academic use permitted, commercial license required
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class CRELayer(nn.Module):
    """
    Single CRE layer with vectorized push-pull cumulative sum.
    
    Achieves O(n) complexity through cumulative operations rather than
    pairwise attention, while maintaining long-range dependency modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        decay_init: float = 0.01,
        dropout: float = 0.1,
        use_push_pull: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_push_pull = use_push_pull
        
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.log_decay = nn.Parameter(
            torch.tensor(math.log(math.exp(decay_init) - 1))
        )
        
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.value.weight, gain=0.01)
        nn.init.zeros_(self.value.bias)
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)
    
    def get_decay_rate(self) -> torch.Tensor:
        return F.softplus(self.log_decay)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        normed = self.norm(x)
        g = torch.sigmoid(self.gate(normed))
        v = self.value(normed)
        
        if self.use_push_pull:
            gv_push = g * v
            gv_pull = (1 - g) * v
            cumsum_push = torch.cumsum(gv_push, dim=1)
            cumsum_pull = torch.cumsum(gv_pull, dim=1)
            cumsum = cumsum_push - cumsum_pull
        else:
            gv = g * v
            cumsum = torch.cumsum(gv, dim=1)
        
        alpha = self.get_decay_rate()
        positions = torch.arange(T, device=x.device, dtype=torch.float32)
        decay = torch.exp(-alpha * positions).view(1, T, 1)
        
        output = cumsum * decay
        out = self.output(output)
        
        return x + self.dropout(out)


class MultiFrequencyCRELayer(nn.Module):
    """
    CRE layer with multiple frequency bands for hierarchical temporal modeling.
    
    Uses different decay rates to capture both short-term and long-term dependencies.
    """
    
    def __init__(
        self,
        d_model: int,
        n_frequencies: int = 3,
        dropout: float = 0.1,
        use_push_pull: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_frequencies = n_frequencies
        self.use_push_pull = use_push_pull
        
        self.norm = nn.LayerNorm(d_model)
        
        self.gates = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_frequencies)
        ])
        self.values = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_frequencies)
        ])
        
        init_decays = [0.1, 0.03, 0.01][:n_frequencies]
        self.log_decays = nn.ParameterList([
            nn.Parameter(torch.tensor(math.log(math.exp(d) - 1)))
            for d in init_decays
        ])
        
        self.output = nn.Linear(d_model * n_frequencies, d_model)
        self.dropout = nn.Dropout(dropout)
        
        for gate in self.gates:
            nn.init.xavier_uniform_(gate.weight)
            nn.init.zeros_(gate.bias)
        for value in self.values:
            nn.init.xavier_uniform_(value.weight, gain=0.01)
            nn.init.zeros_(value.bias)
        nn.init.xavier_uniform_(self.output.weight, gain=0.01)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        normed = self.norm(x)
        
        freq_outputs = []
        for i in range(self.n_frequencies):
            g = torch.sigmoid(self.gates[i](normed))
            v = self.values[i](normed)
            
            if self.use_push_pull:
                gv_push = g * v
                gv_pull = (1 - g) * v
                cumsum_push = torch.cumsum(gv_push, dim=1)
                cumsum_pull = torch.cumsum(gv_pull, dim=1)
                cumsum = cumsum_push - cumsum_pull
            else:
                gv = g * v
                cumsum = torch.cumsum(gv, dim=1)
            
            alpha = F.softplus(self.log_decays[i])
            positions = torch.arange(T, device=x.device, dtype=torch.float32)
            decay = torch.exp(-alpha * positions).view(1, T, 1)
            
            freq_out = cumsum * decay
            freq_outputs.append(freq_out)
        
        combined = torch.cat(freq_outputs, dim=-1)
        out = self.output(combined)
        
        return x + self.dropout(out)


class FeedForward(nn.Module):
    """Standard feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.zeros_(self.net[0].bias)
        nn.init.xavier_uniform_(self.net[3].weight, gain=0.01)
        nn.init.zeros_(self.net[3].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CREBlock(nn.Module):
    """Complete CRE block with resonance layer and feed-forward network."""
    
    def __init__(
        self,
        d_model: int,
        n_frequencies: int = 3,
        dropout: float = 0.1,
        use_push_pull: bool = True,
        use_multi_frequency: bool = True
    ):
        super().__init__()
        
        if use_multi_frequency:
            self.resonance = MultiFrequencyCRELayer(
                d_model, n_frequencies, dropout, use_push_pull
            )
        else:
            self.resonance = CRELayer(d_model, dropout=dropout, use_push_pull=use_push_pull)
        
        self.ff = FeedForward(d_model, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resonance(x)
        x = x + self.ff(self.norm(x))
        return x


class CREModel(nn.Module):
    """
    Complete CRE model for sequence processing.
    
    Stacks multiple CRE blocks with optional embedding layer for token inputs.
    Achieves O(n) complexity in sequence length.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_layers: int = 6,
        n_frequencies: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        use_push_pull: bool = True,
        use_multi_frequency: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([
            CREBlock(d_model, n_frequencies, dropout, use_push_pull, use_multi_frequency)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :T, :]
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden representations before output projection."""
        B, T = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :T, :]
        
        for layer in self.layers:
            x = layer(x)
        
        return self.ln_f(x)


class CREEncoder(nn.Module):
    """
    CRE encoder for sequence representation tasks.
    
    Similar to CREModel but without language modeling head,
    suitable for classification and embedding tasks.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 256,
        n_layers: int = 6,
        n_frequencies: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        use_push_pull: bool = True,
        use_multi_frequency: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([
            CREBlock(d_model, n_frequencies, dropout, use_push_pull, use_multi_frequency)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :T, :]
        
        for layer in self.layers:
            x = layer(x)
        
        return self.ln_f(x)
