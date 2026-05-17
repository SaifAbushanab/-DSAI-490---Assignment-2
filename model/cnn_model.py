"""
Model 3 (out-of-course): 1D CNN Date Generator
─────────────────────────────────────────────────
Architecture:
  - Each of the 4 conditions is embedded independently → (batch, 4, embed_dim)
  - Multiple parallel 1D conv filters with different kernel sizes capture
    different condition interactions (e.g. kernel=2 captures pairs like
    day+month, kernel=3 captures triplets, kernel=4 sees all conditions).
  - Outputs of each conv branch are max-pooled and concatenated.
  - Three classification heads predict dd, mm, yyyy independently.

Using multiple kernel sizes (inspired by TextCNN) lets the model learn both
local condition interactions and global context simultaneously.
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNDateGenerator(nn.Module):
    """
    conditions → (dd_logits, mm_logits, yyyy_logits)

    Parameters
    ----------
    n_days, n_months, n_leaps, n_decades : condition vocab sizes
    embed_dim       : per-condition embedding dimension
    num_filters     : number of filters per kernel size
    kernel_sizes    : list of 1D conv kernel sizes to use in parallel
    dropout         : dropout probability before classification heads
    dd_vocab_size, mm_vocab_size, yyyy_vocab_size : output vocab sizes
    """

    def __init__(
        self,
        n_days: int,
        n_months: int,
        n_leaps: int,
        n_decades: int,
        embed_dim: int,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float,
        dd_vocab_size: int,
        mm_vocab_size: int,
        yyyy_vocab_size: int,
    ) -> None:
        super().__init__()

        self.day_emb    = nn.Embedding(n_days,    embed_dim)
        self.month_emb  = nn.Embedding(n_months,  embed_dim)
        self.leap_emb   = nn.Embedding(n_leaps,   embed_dim)
        self.decade_emb = nn.Embedding(n_decades, embed_dim)

        # Parallel conv branches — one per kernel size.
        # Input to each conv: (batch, embed_dim, 4)   [channels-first]
        # Each conv outputs:  (batch, num_filters, 4 - kernel + 1)
        # After max-pool:     (batch, num_filters)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim,
                          out_channels=num_filters,
                          kernel_size=k),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )
            for k in kernel_sizes
            if k <= 4          # sequence length is 4, skip invalid kernels
        ])

        self.dropout = nn.Dropout(dropout)

        # After concatenating all branches: num_filters * len(valid kernels)
        n_valid  = sum(1 for k in kernel_sizes if k <= 4)
        flat_dim = num_filters * n_valid

        self.dd_head   = nn.Linear(flat_dim, dd_vocab_size)
        self.mm_head   = nn.Linear(flat_dim, mm_vocab_size)
        self.yyyy_head = nn.Linear(flat_dim, yyyy_vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        conditions : LongTensor (batch, 4)

        Returns
        -------
        dd_logits, mm_logits, yyyy_logits
        """
        # Embed each condition: (batch, embed_dim) each
        day_e    = self.day_emb(conditions[:, 0])
        month_e  = self.month_emb(conditions[:, 1])
        leap_e   = self.leap_emb(conditions[:, 2])
        decade_e = self.decade_emb(conditions[:, 3])

        # Stack into sequence: (batch, 4, embed_dim)
        # then transpose to (batch, embed_dim, 4) for Conv1d
        seq = torch.stack([day_e, month_e, leap_e, decade_e], dim=1)
        seq = seq.transpose(1, 2)                          # (B, D, 4)

        # Apply each conv branch and global max-pool
        branch_outputs = []
        for conv in self.convs:
            out = conv(seq)                                # (B, num_filters, L)
            out = F.adaptive_max_pool1d(out, 1).squeeze(-1)  # (B, num_filters)
            branch_outputs.append(out)

        # Concatenate all branches
        h = torch.cat(branch_outputs, dim=-1)              # (B, flat_dim)
        h = self.dropout(h)

        return self.dd_head(h), self.mm_head(h), self.yyyy_head(h)
