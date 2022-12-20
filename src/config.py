SEED = 188
D_MODEL = 768
MAX_SEQ_LEN = 512
SRC_MODEL_NAME = "bert-base-uncased"
TGT_MODEL_NAME = "vinai/phobert-base"

model = dict(
    d_model=D_MODEL,
    pretrained_encoder=SRC_MODEL_NAME,
    # encoder_config=dict(
    #     layer_config=dict(
    #         attention_config=dict(
    #             d_model=D_MODEL,
    #             num_heads=8,
    #             dropout=0.1,
    #         ),
    #         intermediate_config=dict(
    #             d_model=D_MODEL,
    #             d_intermediate=D_MODEL * 4,
    #             dropout=0.1,
    #         ),
    #     ),
    #     pos_encoding_config=dict(
    #         max_seq_len=MAX_SEQ_LEN,
    #         dropout=0.1,
    #     ),
    #     num_layers=6,
    # ),
    decoder_config=dict(
        layer_config=dict(
            attention_config=dict(
                d_model=D_MODEL,
                num_heads=8,
                dropout=0.1,
            ),
            intermediate_config=dict(
                d_model=D_MODEL,
                d_intermediate=D_MODEL * 4,
                dropout=0.1,
            ),
        ),
        pos_encoding_config=dict(
            max_seq_len=MAX_SEQ_LEN,
            dropout=0.1,
        ),
        num_layers=6,
    ),
    adamw_config=dict(
        lr=4e-4,
        finetune_lr=1e-5,
        weight_decay=1e-4,
    ),
    scheduler_config=dict(
        gamma=0.9,
    ),
)
