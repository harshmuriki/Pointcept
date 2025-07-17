"""
PTv3 + PPT adapted for Peachtree Pruning Dataset
Binary segmentation for peach tree pruning (keep vs prune) with tree type classification.
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 4  # Reduced batch size to help with debugging
num_worker = 8  # Reduced workers to help with debugging
mix_prob = 0.8
empty_cache = False
enable_amp = False  # Disable AMP for easier debugging
find_unused_parameters = True
clip_grad = 1.0  # Reduced gradient clipping

# trainer
train = dict(
    type="DefaultTrainer",  # Use default trainer instead of multi-dataset
)

# model settings
model = dict(
    type="DefaultSegmentorV2",  # Use V2 to match base config
    num_classes=2,  # Binary classification: keep (0) vs prune (1)
    backbone_out_channels=64,  # Add this to match base config
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,  # coord (3) + color (3) + normal (3) - matching base config
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),  # 5 stages: len(stride) + 1
        enc_channels=(32, 64, 128, 256, 512),  # 5 channels for 5 stages
        enc_num_head=(2, 4, 8, 16, 32),  # 5 heads for 5 stages
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),  # 5 patch sizes
        dec_depths=(2, 2, 2, 2),  # 4 decoder stages
        dec_channels=(64, 64, 128, 256),  # 4 decoder channels
        dec_num_head=(4, 4, 8, 16),  # 4 decoder heads
        dec_patch_size=(1024, 1024, 1024, 1024),  # 4 decoder patch sizes
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
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("PeachTree",),  # Single condition for peach tree
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    # fmt: off
    # class_name=(
    #     "keep",    # class 0: points to keep
    #     "prune",   # class 1: points to prune
    # ),
    # valid_index=(
    #     (0, 1),
    # ),
    # # fmt: on
    # backbone_mode=False,
)

# scheduler settings
epoch = 200  # Reduced epochs for smaller dataset
optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.02)  # Adjusted learning rate
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.003,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
# dataset settings
dataset_type = "PeachTreeDataset"
data_root = "data/peachtreev3_2"

data = dict(
    num_classes=2,
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
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout",
                dropout_ratio=0.1,
                dropout_application_ratio=0.1,
            ),
            # Tree-specific augmentations
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.002, clip=0.01),
            dict(
                type="ElasticDistortion",
                distortion_params=[[0.1, 0.2], [0.4, 0.8]],
            ),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.005,  # Smaller grid size for tree detail
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=50000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "tree_type"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
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
                grid_size=0.005,
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
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "tree_type",
                    "origin_segment",
                    "inverse",
                ),
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
                grid_size=0.005,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
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
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.98, 0.98]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.98, 0.98]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.98, 0.98]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.98, 0.98]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.02, 1.02]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.02, 1.02]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.02, 1.02]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.02, 1.02]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

# hooks - for loading pretrained weights with different number of classes
# hooks = [
#     dict(type="CheckpointLoader", strict=False),  # Allow loading pretrained weights with mismatched layer sizes
#     dict(type="IterationTimer", warmup_iter=2),
#     dict(type="InformationWriter"),
#     dict(type="SemSegEvaluator"),
#     dict(type="CheckpointSaver", save_freq=None),
#     dict(type="PreciseEvaluator", test_last=False),
# ]
