"""
PTv3 + PPT adapted for Peachtree Pruning Dataset
Binary segmentation for peach tree pruning (keep vs prune) with tree type classification.

IMPORTANT: This configuration is set up to fine-tune a pretrained PPT (Point Prompt Training) model.
To use this config:
1. Download/obtain a pretrained PPT model with 36 classes and 3 conditions
2. Update the 'weight' parameter below to point to your pretrained model file
3. The model will use only classes 0 (keep) and 1 (prune) for the ScanNet condition
4. PeachTree data is mapped to use the "ScanNet" condition with binary segmentation
5. Classes 2-35 are placeholders to maintain compatibility with the pretrained model
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4  # Reduced batch size to help with debugging
num_worker = 8  # Reduced workers to help with debugging
mix_prob = 0.8
empty_cache = False
enable_amp = False  # Disable AMP for easier debugging
find_unused_parameters = True
clip_grad = 3.0  # Reduced gradient clipping

# trainer
train = dict(
    type="MultiDatasetTrainer",  # Use MultiDatasetTrainer with ConcatDataset
)

# model settings
model = dict(
    type="PPT-v1m1",  # Use PPT model type to match the pretrained model
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,  # coord (3) + color (3) + normal (3) - matching base config
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 6, 3),  # Match original model architecture
        enc_channels=(48, 96, 192, 384, 512),  # Match original model architecture
        enc_num_head=(3, 6, 12, 24, 32),  # Match original model architecture
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),  # Match original model architecture
        dec_channels=(64, 96, 192, 384),  # Match original model architecture
        dec_num_head=(4, 6, 12, 24),  # Match original model architecture
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=True,  # Match original model
        pdnorm_ln=True,  # Match original model
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("Structured3D", "ScanNet", "S3DIS"),  # Match pretrained model conditions
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=64,  # Match original model
    context_channels=256,  # Match original model
    conditions=("Structured3D", "ScanNet", "S3DIS"),  # Match pretrained model conditions
    template="[x]",  # Match original model template
    clip_model="ViT-B/16",  # Match original model
    class_name=(
        "keep",    # class 0: points to keep
        "prune",   # class 1: points to prune
        # Classes 2-35: unused placeholder classes for PPT compatibility
        "unused_2", "unused_3", "unused_4", "unused_5", "unused_6", "unused_7", "unused_8", "unused_9",
        "unused_10", "unused_11", "unused_12", "unused_13", "unused_14", "unused_15", "unused_16", "unused_17",
        "unused_18", "unused_19", "unused_20", "unused_21", "unused_22", "unused_23", "unused_24", "unused_25",
        "unused_26", "unused_27", "unused_28", "unused_29", "unused_30", "unused_31", "unused_32", "unused_33",
        "unused_34", "unused_35",
    ),
    valid_index=(
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),  # Structured3D
        (0, 1),  # ScanNet - modified for PeachTree binary segmentation (keep vs prune)
        (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),  # S3DIS
    ),
    backbone_mode=False,
)

# scheduler settings
eval_epoch = epoch = 200  # Match original model epoch count
param_dicts = [dict(keyword='block', lr=0.0005)]  # Match original model param dicts
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.05)  # Match original model optimizer
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.005, 0.0005],  # Match original model scheduler
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "PeachTreeDataset"
data_root = "data/peachtreev3_1"

data = dict(
    num_classes=2,  # PPT model will output 2 classes based on valid_index, not 36
    ignore_index=-1,
    names=[
        "keep",    # class 0: points to keep
        "prune",   # class 1: points to prune
    ],
    tree_type_names=[
        "type_0",  # tree type 0
        "type_1",  # tree type 1
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                split=["train", "val", "test"],
                data_root=data_root,
                transform=[
                    dict(type="CenterShift", apply_z=True),
                    dict(
                        type="RandomDropout",
                        dropout_ratio=0.2,
                        dropout_application_ratio=0.2,
                    ),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
                    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    dict(
                        type="ElasticDistortion",
                        distortion_params=[[0.2, 0.4], [0.8, 1.6]],
                    ),
                    dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
                    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
                    dict(type="ChromaticJitter", p=0.95, std=0.05),
                    # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
                    # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
                    dict(
                        type="GridSample",
                        grid_size=0.02,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    dict(type="SphereCrop", sample_rate=0.8, mode="random"),
                    dict(type="SphereCrop", point_max=204800, mode="random"),
                    dict(type="CenterShift", apply_z=False),
                    dict(type="NormalizeColor"),
                    # dict(type="ShufflePoint"),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("color", "normal"),
                    ),
                ],
                test_mode=False,
                loop=3,  # Add loop parameter for consistency
            ),
        ],
        loop=1,  # Add loop parameter for ConcatDataset
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "tree_type", "origin_segment", "inverse", "condition"),  # Include condition
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="Update", keys_dict={"condition": "ScanNet"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),  # Include condition
                    feat_keys=("color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[0],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[0.98, 0.98]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[0.98, 0.98]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[0.98, 0.98]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[0.98, 0.98]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[0],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[1.02, 1.02]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[1.02, 1.02]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[1.02, 1.02]),
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     ),
                #     dict(type="RandomScale", scale=[1.02, 1.02]),
                # ],
                # [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

# hooks - for loading pretrained PPT weights with selective loading
hooks = [
    dict(type="CheckpointLoader", strict=False),  # Try strict=False first
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
