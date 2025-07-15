_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 8  # bs: total bs in all gpus (reduced for smaller dataset)
mix_prob = 0.8
empty_cache = False
enable_amp = True

# model settings
model = dict(
    type="DefaultSegmentorWithTreeType",
    backbone=dict(
        type="PT-v2m2",
        in_channels=9,  # coord (3) + color (3) + normal (3)
        num_classes=2,  # binary classification: keep (0) vs prune (1)
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.02, 0.05, 0.125, 0.3125),  # Smaller grid sizes for tree-scale data
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        enable_checkpoint=False,
        unpool_backend="map",  # map / interp
    ),
    # Multi-task learning setup
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),  # Segmentation loss
        dict(type="CrossEntropyLoss", loss_weight=0.5, ignore_index=-1),  # Tree type loss
    ],
    # Tree type classification head
    tree_type_head=dict(
        type="TreeTypeHead",
        in_channels=512,  # Match the final encoder channels
        num_classes=2,    # Binary tree type classification
        dropout_ratio=0.1,
    ),
)

# scheduler settings
epoch = 300  # Fewer epochs for smaller dataset
optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.02)  # Slightly lower learning rate
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "PeachTreeDataset"  # Use custom peach tree dataset
data_root = "data/peachtreev2"  # Use the standard scannet data path

data = dict(
    num_classes=2,
    ignore_index=-1,
    names=[
        "keep",    # class 0: points to keep
        "prune",   # class 1: points to prune
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.1, dropout_application_ratio=0.1
            ),  # Reduced dropout for smaller dataset
            # Tree-specific augmentations
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 32, 1 / 32], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),  # Smaller scale variation
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.002, clip=0.01),  # Smaller jitter for tree scale
            dict(type="ElasticDistortion", distortion_params=[[0.1, 0.2], [0.4, 0.8]]),  # Reduced distortion
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.005,  # Smaller grid size for tree detail
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=50000, mode="random"),  # Smaller point limit
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "tree_type"),  # Include tree_type
                feat_keys=("coord", "color", "normal"),  # Include normals for better geometry
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
            dict(
                type="GridSample",
                grid_size=0.005,  # Same as training
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment", "tree_type"),  # Include tree_type
                feat_keys=("coord", "color", "normal"),  # Include normals for better geometry
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
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "color", "normal"),  # Include normals for better geometry
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
