from .transformer import Encoder, Decoder
from utils.scheduler import WarmUpScheduler

from typing import Optional
import math
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
        pretrained_encoder: str,
        decoder_config: dict,
        d_model: int,
        adamw_config: dict,
        scheduler_config: dict,
    ):
        super().__init__()

        self.adamw_config = adamw_config
        self.scheduler_config = scheduler_config
        self.d_model = d_model

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_embedding = nn.Embedding(
            num_embeddings=src_tokenizer.vocab_size,
            embedding_dim=d_model,
            padding_idx=src_tokenizer.pad_token_id,
        )

        self.tgt_embedding = nn.Embedding(
            num_embeddings=tgt_tokenizer.vocab_size,
            embedding_dim=d_model,
            padding_idx=tgt_tokenizer.pad_token_id,
        )

        self.transformer = nn.Transformer(self.d_model, nhead=8, batch_first=True)

        self.linear = nn.Linear(d_model, tgt_tokenizer.vocab_size)

        self.loss = MeanMetric()

        self.save_hyperparameters(ignore=["src_tokenizer", "tgt_tokenizer"])

    def forward(
        self,
        src_token_ids: torch.Tensor,
        tgt_token_ids: torch.Tensor,
        src_attention_mask: Optional[torch.BoolTensor] = None,
        tgt_attention_mask: Optional[torch.BoolTensor] = None,
    ):
        src_embedding = self.src_embedding(src_token_ids) * math.sqrt(self.d_model)
        tgt_embedding = self.tgt_embedding(tgt_token_ids) * math.sqrt(self.d_model)

        decoder_output = self.transformer(
            src_embedding,
            tgt_embedding,
            src_key_padding_mask=(~src_attention_mask),
            tgt_key_padding_mask=(~tgt_attention_mask),
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

        predicted_token_ids = torch.argmax(logits, dim=-1)
        acc = (predicted_token_ids == tgt_out) * tgt_out_mask
        acc = acc.sum() / tgt_out_mask.sum()

        return acc

    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack(outputs).mean()
        self.log("val/acc", avg_acc, on_epoch=True)

    def configure_optimizers(self):
        param_groups = [
            {
                "params": self.encoder.parameters(),
                "lr": self.adamw_config["finetune_lr"],
            },
            {"params": self.decoder.parameters(), "lr": self.adamw_config["lr"]},
            {"params": self.linear.parameters(), "lr": self.adamw_config["lr"]},
        ]

        optimizer = torch.optim.Adam(
            param_groups, weight_decay=self.adamw_config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, **self.scheduler_config
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def greedy_translate(self, text: str, max_translation_length: int = 100):
        src_token_ids, src_attention_mask = self.src_tokenizer(
            text, return_token_type_ids=False, return_tensors="pt"
        ).values()

        src_token_ids = src_token_ids.to(self.device)
        src_attention_mask = src_attention_mask.to(self.device)

        tgt_token_ids = torch.tensor(
            [[self.tgt_tokenizer.cls_token_id]], device=self.device
        )
        tgt_attention_mask = torch.tensor([[1]], device=self.device)

        for _ in range(max_translation_length):
            logits = self(
                src_token_ids, tgt_token_ids, src_attention_mask, tgt_attention_mask
            )

            next_tgt_token_id = torch.argmax(logits[:, -1, :], keepdim=True, dim=-1)
            tgt_token_ids = torch.cat([tgt_token_ids, next_tgt_token_id], dim=-1)
            tgt_attention_mask = torch.cat(
                [
                    tgt_attention_mask,
                    torch.ones_like(next_tgt_token_id, dtype=torch.int64)
                    if next_tgt_token_id != self.tgt_tokenizer.pad_token_id
                    else torch.zeros_like(next_tgt_token_id, dtype=torch.int64),
                ],
                dim=-1,
            )

            if next_tgt_token_id == self.tgt_tokenizer.sep_token_id:
                break

        return self.tgt_tokenizer.decode(tgt_token_ids[0], skip_special_tokens=True)
