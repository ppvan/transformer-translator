from model import Translator
from dataset import Iwlst2015DataModule
import config

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from transformers import AutoTokenizer


def main():
    src_tokenizer = AutoTokenizer.from_pretrained(config.SRC_MODEL_NAME)
    tgt_tokenizer = AutoTokenizer.from_pretrained(config.TGT_MODEL_NAME)

    model = Translator(src_tokenizer, tgt_tokenizer, **config.model)

    data_module = Iwlst2015DataModule(
        src_tokenizer, tgt_tokenizer, config.MAX_SEQ_LEN, config.trainer["batch_size"]
    )

    trainer = Trainer(
        accelerator=config.trainer["accelerator"],
        max_epochs=config.trainer["max_epochs"],
        callbacks=[
            ModelCheckpoint(config.trainer["ckpt_dir"], monitor="val/loss", mode="min"),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=WandbLogger(
            project="Machine Translation", name="Transformer"
        ),
        gradient_clip_val=config.trainer["gradient_clip_val"]
    )

    trainer.fit(model, data_module, ckpt_path=config.trainer["ckpt_path"])

if __name__ == "__main__":
    seed_everything(config.SEED)
    main()
