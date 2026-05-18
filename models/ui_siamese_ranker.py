# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class UISiameseRanker(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=256, out_dim=128):
        super().__init__()

        self.text_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, out_dim),
        )

        self.ui_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_query(self, query_vec):
        z = self.text_encoder(query_vec)
        return F.normalize(z, dim=-1)

    def encode_ui(self, ui_vec):
        z = self.ui_encoder(ui_vec)
        return F.normalize(z, dim=-1)

    def forward(self, query_vec, ui_vec):
        q = self.encode_query(query_vec)
        u = self.encode_ui(ui_vec)
        return (q * u).sum(dim=-1)
