import math
import torch
from torch import nn

from .modules import MultiHeadAttention, Intermediate, PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, attention_config: dict, intermediate_config: dict):
        super().__init__()

        self.attention = MultiHeadAttention(**attention_config)
        self.intermediate = Intermediate(**intermediate_config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        x, _ = self.attention(x, x, x, attention_mask=attention_mask)
        x = self.intermediate(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, attention_config: dict, intermediate_config: dict):
        super().__init__()

        self.enc_attention = MultiHeadAttention(**attention_config)
        self.self_attention = MultiHeadAttention(**attention_config)
        self.intermediate = Intermediate(**intermediate_config)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        cross_attention_mask: torch.Tensor = None,
        self_attention_mask: torch.Tensor = None,
    ):
        x, _ = self.self_attention(x, x, x, attention_mask=self_attention_mask)

        x, _ = self.enc_attention(
            x, encoder_output, encoder_output, attention_mask=cross_attention_mask
        )
        x = self.intermediate(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        layer_config: dict,
        pos_encoding_config: dict,
        num_layers: int,
        d_model: int,
        vocab_size: int,
        padding_idx: int,
    ):
        """
        Args:
            x (LongTensor). Shape (batch_size, seq_len)
            attention_mask (BoolTensor): The attention mask of sequences.
            `False` value of the mask denote the special token which will be ignored in computing attention process.
            Shape (batch_size, seq_len).

        Returns:
            out (FloatTensor). Shape (batch_size, seq_len, d_model)
        """
        super().__init__()

        self.vocab_size = vocab_size

        self.pos_encoding = PositionalEncoding(d_model=d_model, **pos_encoding_config)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx
        )

        self.layers = nn.ModuleList(
            [EncoderLayer(**layer_config) for _ in range(num_layers)]
        )

    @staticmethod
    def _create_attention_mask(
        src_attention_mask: torch.Tensor, tgt_attention_mask: torch.Tensor
    ):
        """
        Args:
            src_attention_mask (Tensor). Shape (batch_size, src_seq_len) \n
            tgt_attention_mask (Tensor). Shape (batch_size, tgt_seq_len)

        Returns:
            attention_mask (Tensor). Shape (batch_size, src_seq_len, tgt_seq_len)
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, seq_len)
        attention_mask = torch.minimum(
            src_attention_mask.unsqueeze(2), tgt_attention_mask.unsqueeze(1)
        )

        return attention_mask

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        # (batch_size, seq_len, d_model)
        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoding(x)

        if attention_mask is not None:
            attention_mask = self._create_attention_mask(attention_mask, attention_mask)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        layer_config: dict,
        pos_encoding_config: dict,
        num_layers: int,
        d_model: int,
        vocab_size: int,
        padding_idx: int,
    ):
        """
        Args:
            x (LongTensor). Shape (batch_size, seq_len)
            encoder_output (FloatTensor). Shape (batch_size, seq_len, d_model)
            src_attention_mask (BoolTensor): The attention mask of source sequences.
            `False` value of the mask denote the special token which will be ignored in computing attention process.
            Shape (batch_size, seq_len).
            tar_attention_mask (BoolTensor): The attention mask of target sequences.
            `False` value of the mask denote the special token which will be ignored in computing attention process.
            Shape (batch_size, seq_len).

        Returns:
            out (FloatTensor). Shape (batch_size, seq_len, d_model)
        """
        super().__init__()

        self.vocab_size = vocab_size

        self.pos_encoding = PositionalEncoding(d_model=d_model, **pos_encoding_config)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(**layer_config) for _ in range(num_layers)]
        )

    @staticmethod
    def _create_attention_mask(
        src_attention_mask: torch.Tensor, tgt_attention_mask: torch.Tensor
    ):
        """
        Args:
            src_attention_mask (Tensor). Shape (batch_size, src_seq_len) \n
            tgt_attention_mask (Tensor). Shape (batch_size, tgt_seq_len)

        Returns:
            attention_mask (Tensor). Shape (batch_size, src_seq_len, tgt_seq_len)
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, seq_len)
        attention_mask = torch.minimum(
            src_attention_mask.unsqueeze(2), tgt_attention_mask.unsqueeze(1)
        )

        return attention_mask

    @staticmethod
    def _create_subsequent_mask(seq_len: int, device: torch.device):
        """
        Returns:
            subsequent_mask (Tensor). Shape (1, seq_len, seq_len)
        """
        subsequent_mask = 1 - torch.triu(
            torch.ones((seq_len, seq_len), device=device), diagonal=1
        )

        return subsequent_mask.unsqueeze_(0)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        enc_attention_mask: torch.Tensor = None,
        dec_attention_mask: torch.Tensor = None,
    ):
        # (batch_size, seq_len, d_model)
        x = self.embedding(x) * math.sqrt(x.size(-1))
        x = self.pos_encoding(x)

        # create cross attention mask
        if enc_attention_mask is not None and dec_attention_mask is not None:
            cross_attention_mask = self._create_attention_mask(
                dec_attention_mask, enc_attention_mask
            )
        elif enc_attention_mask is not None:
            cross_attention_mask = enc_attention_mask.unsqueeze(1).repeat_interleave(
                x.size(1), dim=1
            )
        elif dec_attention_mask is not None:
            cross_attention_mask = dec_attention_mask.unsqueeze(2).repeat_interleave(
                encoder_output.size(1), dim=2
            )
        else:
            cross_attention_mask = None

        # create self attention mask
        subsequent_mask = self._create_subsequent_mask(x.size(1), x.device)

        if dec_attention_mask is not None:
            self_attention_mask = torch.minimum(
                self._create_attention_mask(dec_attention_mask, dec_attention_mask),
                subsequent_mask,
            )
        else:
            self_attention_mask = subsequent_mask

        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                cross_attention_mask=cross_attention_mask,
                self_attention_mask=self_attention_mask,
            )

        return x
