from .transformer import Encoder, Decoder
from utils.scheduler import WarmUpScheduler

from typing import Optional
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import PreTrainedTokenizer


class Translator(LightningModule):
    def __init__(
        self,
        src_tokenizer: PreTrainedTokenizer,
        tgt_tokenizer: PreTrainedTokenizer,
        encoder_config: dict,
        decoder_config: dict,
        d_model: int,
        adamw_config: dict,
        warmup_scheduler_config: dict,
    ):
        super().__init__()

        self.adamw_config = adamw_config
        self.warmup_scheduler_config = warmup_scheduler_config
        self.d_model = d_model

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.encoder = Encoder(
            vocab_size=src_tokenizer.vocab_size,
            padding_idx=src_tokenizer.pad_token_id,
            d_model=d_model,
            **encoder_config
        )
        self.decoder = Decoder(
            vocab_size=tgt_tokenizer.vocab_size,
            padding_idx=tgt_tokenizer.pad_token_id,
            d_model=d_model,
            **decoder_config
        )

        self.linear = nn.Linear(d_model, tgt_tokenizer.vocab_size)

        self.loss = MeanMetric()

        self.save_hyperparameters(ignore=["src_tokenizer", "tgt_tokenizer"])

    def forward(
        self,
        src_token_ids: torch.Tensor,
        tgt_token_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        tgt_attention_mask: Optional[torch.Tensor] = None,
    ):
        encoder_output = self.encoder(src_token_ids, src_attention_mask)
        decoder_output = self.decoder(
            tgt_token_ids, encoder_output, src_attention_mask, tgt_attention_mask
        )

        logits = self.linear(decoder_output)

        return logits

    def _compute_loss(self, logits, target, tgt_mask):
        logits = torch.flatten(logits, end_dim=-2)
        target = torch.flatten(target)
        tgt_mask = torch.flatten(tgt_mask)

        loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
        loss = loss * tgt_mask
        loss = loss.sum() / tgt_mask.sum()

        return loss

    def training_step(self, batch, batch_idx):
        tgt_in = batch["tgt"]["input_ids"][:, :-1]
        tgt_in_mask = batch["tgt"]["attention_mask"][:, :-1]
        tgt_out = batch["tgt"]["input_ids"][:, 1:]
        tgt_out_mask = batch["tgt"]["attention_mask"][:, 1:]

        logits = self(
            batch["src"]["input_ids"],
            tgt_in,
            batch["src"]["attention_mask"],
            tgt_in_mask,
        )

        loss = self._compute_loss(logits, tgt_out, tgt_out_mask)

        # logging
        self.loss(loss)

        if batch_idx % 200 == 0:
            self.log("train/loss", self.loss, on_epoch=True, on_step=True)
            self.loss.reset()

        return loss

    def validation_step(self, batch, batch_idx):
        tgt_in = batch["tgt"]["input_ids"][:, :-1]
        tgt_in_mask = batch["tgt"]["attention_mask"][:, :-1]
        tgt_out = batch["tgt"]["input_ids"][:, 1:]
        tgt_out_mask = batch["tgt"]["attention_mask"][:, 1:]

        logits = self(
            batch["src"]["input_ids"],
            tgt_in,
            batch["src"]["attention_mask"],
            tgt_in_mask,
        )

        loss = self._compute_loss(logits, tgt_out, tgt_out_mask)

        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("val/loss", avg_loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.adamw_config)
        scheduler = WarmUpScheduler(
            optimizer, feature_size=self.d_model, **self.warmup_scheduler_config
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
