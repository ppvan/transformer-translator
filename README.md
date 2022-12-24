# Transformer for Machine Translation

This is a PyTorch implementation of the Transformer model described in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The model is trained to translate from English to Vietnamese on the [IWSLT 2015 dataset](https://sites.google.com/site/iwsltevaluation2015/mt-track?pli=1)

Since limited GPU resources are available, I use pretrained BERT base as the encoder.

## Usage
First, install the dependencies:
```
pip install -r requirements.txt
```

Then, run the training script:
```bash
python3 src/train.py \
    --accelerator gpu
```

Or if you just want to try it out, you can download the checkpoint with following command:
```bash
#!/bin/bash
fileid="1K6ULeDN4DIEdMF-Vo5EmLA4wHj4s7dfK"
filename="checkpoints/last.ckpt"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
```

After training, you can run the demo script to translate a sentence:
```bash
python3 src/test.py --ckpt_path <path to checkpoint>
```
where: 
- `<path to checkpoint>` is the path to the checkpoint file you want to use. Default is `./checkpoints/last.ckpt`

![Demo](./assets/demo.gif)