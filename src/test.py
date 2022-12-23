import config
from model import Translator

from transformers import AutoTokenizer
from argparse import ArgumentParser


def main():
    argparse = ArgumentParser()
    argparse.add_argument("--ckpt_path", type=str, default="checkpoints/last-v1.ckpt")
    argparse.add_argument("--beam_size", type=int, default=0)
    argparse.add_argument("--max_len", type=int, default=50)

    args = argparse.parse_args()

    src_tokenizer = AutoTokenizer.from_pretrained(config.SRC_MODEL_NAME)
    tgt_tokenizer = AutoTokenizer.from_pretrained(config.TGT_MODEL_NAME)

    model = Translator.load_from_checkpoint(
        args.ckpt_path,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )

    while True:
        src = input("Enter source text: ")

        if args.beam_size > 0:
            tgt = model.beam_translate(
                src, max_translation_length=args.max_len, beam_size=args.beam_size
            )
        else:
            tgt = model.greedy_translate(src, max_translation_length=args.max_len)

        print(f"Translated text: {tgt}")


if __name__ == "__main__":
    main()
