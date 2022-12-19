import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout: float = 0.1):
        """
        Args:
            x (Tensor): The input tensor. Shape (batch_size, seq_len, d_model)
        """
        super().__init__()

        self.register_buffer("encoding_table", self.get_encoding_table(max_seq_len, d_model))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def get_encoding_table(max_seq_len: int, d_model: int):
        pos = torch.arange(max_seq_len).unsqueeze_(1)
        d_model_ids = torch.arange(d_model).unsqueeze_(0)

        enc_table = pos / torch.pow(10000, 2 * d_model_ids / d_model)

        enc_table[:, 0::2] = torch.sin(enc_table[:, 0::2])
        enc_table[:, 1::2] = torch.cos(enc_table[:, 1::2])

        return enc_table.unsqueeze_(0)  # dim for batch

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)

        x = x + self.encoding_table[:, :seq_len, :]
        x = self.dropout(x)

        return x
