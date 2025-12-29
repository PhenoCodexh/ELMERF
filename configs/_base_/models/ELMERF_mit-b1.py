# model settings
custom_imports = dict(
    imports=['mmseg.engine.hooks.dice_weight_logger'],
    allow_failed_imports=False
)

norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='Edge_LoG_MixVisionTransformer_cfg',
        # Transformer
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,

        # CNN
        cnn_cfg=dict(
            in_channels=1,
            denoise_cfg=dict(
                out_channels=64,  # Number of channels after noise reduction.
                norm_groups=8,  # GroupNorm number of groups
                se_reduction=16
            ),
            dims=(64, 128),
            depths=(2, 4),
            drop_path=0.1,
            conv_cfg=dict(
                layer_scale_init_value=1e-6,  # Initial zoom level
                expan_ratio=4,  # FFN expansion ratio
                kernel_size=7  # Depthwise convolution kernel size
            ),
            se_reduction=16
        )
    ),

    decode_head=dict(
        type='ERFHead',
        in_index=[0, 1, 2, 3,4,5],
        in_channels=[32, 64, 160, 256,64,128],
        feature_strides=[4, 8, 16, 32,4,8],
        channels=256,
        num_classes=4,
        decoder_params=dict(embed_dim=256),
ignore_index=255,
        loss_decode=
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,class_weight=[1.0,5.0,5.5,5.4]),
        ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)