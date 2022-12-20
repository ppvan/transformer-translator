from model import Translator
from dataset import Iwlst2015DataModule
import config

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from transformers import AutoTokenizer


def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--accelerator", type=str, default="gpu")

    args = parser.parse_args()

    src_tokenizer = AutoTokenizer.from_pretrained(config.SRC_MODEL_NAME)
    tgt_tokenizer = AutoTokenizer.from_pretrained(config.TGT_MODEL_NAME)

    model = Translator(src_tokenizer, tgt_tokenizer, **config.model)

    data_module = Iwlst2015DataModule(
        src_tokenizer, tgt_tokenizer, config.MAX_SEQ_LEN, args.batch_size
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        max_epochs=args.max_epochs,
        callbacks=[
            ModelCheckpoint(args.ckpt_dir, monitor="val/loss", mode="min", save_last=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=WandbLogger(project="Machine Translation", name="Transformer", version=0),
        gradient_clip_val=args.gradient_clip_val,
    )

    trainer.fit(model, data_module, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    seed_everything(config.SEED)
    main()
