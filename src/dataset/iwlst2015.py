import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from transformers import PreTrainedTokenizer
from datasets import load_dataset

from utils.text_processing import fix_contents


class Iwlst2015DataModule(LightningDataModule):
    def __init__(
        self,
        en_tokenizer: PreTrainedTokenizer,
        vi_tokenizer: PreTrainedTokenizer,
        max_length: int,
        batch_size: int = 8,
        num_workers: int = 2,
    ):
        super().__init__()

        self.en_tokenizer = en_tokenizer
        self.vi_tokenizer = vi_tokenizer

        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        data_path = "mt_eng_vietnamese"
        data_name = "iwslt2015-vi-en"

        self.train_dataset = load_dataset(data_path, data_name, split="test")
        self.val_dataset = load_dataset(data_path, data_name, split="validation")
        # self.train_dataset = load_dataset(data_path, data_name, split="test")

        self.train_dataset = self._preprocessing_and_tokenize(self.train_dataset)
        self.val_dataset = self._preprocessing_and_tokenize(self.val_dataset)
        # self.train_dataset = self._preprocessing_and_tokenize(self.train_dataset)

    def _preprocessing_and_tokenize(self, dataset):
        dataset.set_format(type="torch")

        def tokenize(item):
            # preprocess
            item["translation"]["en"] = fix_contents(item["translation"]["en"])

            item["en"] = self.en_tokenizer(
                item["translation"]["en"],
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                return_tensors="pt",
            )

            item["vi"] = self.vi_tokenizer(
                item["translation"]["vi"],
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
                return_tensors="pt",
            )

            del item["translation"]

            return item

        dataset = dataset.map(tokenize)

        return dataset

    def collate_fn(self, batch):
        ret = {
            "src": {
                "input_ids": [],
                "attention_mask": [],
            },
            "tgt": {
                "input_ids": [],
                "attention_mask": [],
            },
        }

        max_src_len = max(item["en"]["input_ids"].size(1) for item in batch)
        max_tgt_len = max(item["vi"]["input_ids"].size(1) for item in batch)

        for item in batch:
            input_ids = item["en"]["input_ids"]
            attention_mask = item["en"]["attention_mask"]

            # padding to the left of the sequence to match the max length
            ret["src"]["input_ids"].append(
                F.pad(
                    input_ids,
                    (max_src_len - input_ids.size(1), 0),
                    value=self.en_tokenizer.pad_token_id,
                )
            )

            ret["src"]["attention_mask"].append(
                F.pad(
                    attention_mask,
                    (max_src_len - attention_mask.size(1), 0),
                    value=0,
                )
            )

            input_ids = item["vi"]["input_ids"]
            attention_mask = item["vi"]["attention_mask"]

            ret["tgt"]["input_ids"].append(
                F.pad(
                    input_ids,
                    (max_tgt_len - input_ids.size(1), 0),
                    value=self.vi_tokenizer.pad_token_id,
                )
            )

            ret["tgt"]["attention_mask"].append(
                F.pad(
                    attention_mask,
                    (max_tgt_len - attention_mask.size(1), 0),
                    value=0,
                )
            )

        ret["src"]["input_ids"] = torch.cat(ret["src"]["input_ids"], dim=0)
        ret["src"]["attention_mask"] = torch.cat(ret["src"]["attention_mask"], dim=0)
        ret["tgt"]["input_ids"] = torch.cat(ret["tgt"]["input_ids"], dim=0)
        ret["tgt"]["attention_mask"] = torch.cat(ret["tgt"]["attention_mask"], dim=0)

        return ret

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
