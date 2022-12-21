import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        """
        Multi-Head Attention module

        Args:
            q (Tensor): The query. Shape (batch_size, src_seq_len, d_model)
            k (Tensor): The key. Shape (batch_size, dst_seq_len, d_model)
            v (Tensor): The value. Shape (batch_size, dst_seq_len, d_model)
            attention_mask (BoolTensor): The attention mask of query.
                                    The token corresponding with value `False` in the mask will be ignored in computing attention process. \
                                    Shape (batch_size, src_seq_len, dst_seq_len)

        Returns:
            context (Tensor): The context vector. Shape (batch_size, src_seq_len, d_model)
        """
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self._attn_scale_factor = math.sqrt(self.head_dim)

        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)

        self.proj_o = nn.Linear(d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)

    def _split_heads(self, x: torch.Tensor):
        # x.shape == (batch_size, seq_len, d_model)
        x = x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).contiguous()

        # x.shape == (batch_size, num_heads, seq_len, head_dim)
        x = x.transpose(1, 2)

        return x

    def _combine_heads(self, x: torch.Tensor):
        # x.shape == (batch_size, num_heads, seq_len, head_dim)
        x = x.transpose(1, 2).contiguous()

        # x.shape == (batch_size, seq_len, d_model)
        x = x.view(x.size(0), x.size(1), self.d_model)

        return x

    def _get_query_key_mask(self, mask: torch.BoolTensor) -> torch.BoolTensor:
        # reshape to (batch_size, num_heads, src_seq_len, dst_seq_len)
        if len(mask.shape) == 3:
            return mask.unsqueeze(1)
        else:
            raise ValueError(
                "Mask shape must be (batch_size, src_seq_len, dst_seq_len)"
            )

    def _dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        # (batch_size, num_heads, src_seq_len, dst_seq_len)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / self._attn_scale_factor

        if mask is not None:
            mask = self._get_query_key_mask(mask)
            attn_scores = attn_scores + (1 - mask) * -1e9

        # stable softmax
        attn_probs = torch.softmax(
            attn_scores - attn_scores.max(-1, keepdim=True).values, dim=-1
        )
        attn_probs = self.dropout(attn_probs)
        attn_outputs = torch.matmul(attn_probs, v)

        return attn_outputs, attn_probs

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        residual = q

        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_outputs, attn_probs = self._dot_product_attention(q, k, v, attention_mask)
        attn_outputs = self._combine_heads(attn_outputs)

        outputs = self.proj_o(attn_outputs)
        outputs = self.layernorm(residual + self.dropout(outputs))

        return outputs, attn_probs
