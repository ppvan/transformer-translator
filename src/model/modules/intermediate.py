import torch
from torch import nn


class Intermediate(nn.Module):
    def __init__(self, d_model: int, d_intermediate: int, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_intermediate = d_intermediate

        self.proj_i = nn.Linear(d_model, d_intermediate)
        self.proj_o = nn.Linear(d_intermediate, d_model)

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x

        x = self.proj_i(x)
        x = self.activation(x)
        x = self.proj_o(x)

        x = self.dropout(x)
        x = self.layernorm(residual + x)

        return x
