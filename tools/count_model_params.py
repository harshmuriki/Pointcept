#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/vmuriki3/Documents/transformer/peachtree-pruning/PointTransformerV3/Pointcept')

import torch
from pointcept.models import build_model
from pointcept.utils.config import Config

def count_model_parameters(config_path):
    """Count parameters in the PointTransformer V2 model"""
    
    print("=" * 60)
    print("POINTTRANSFORMER V2 MODEL PARAMETER ANALYSIS")
    print("=" * 60)
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    print(f"Config file: {config_path}")
    print(f"Model type: {cfg.model.backbone.type}")
    print(f"Input channels: {cfg.model.backbone.in_channels}")
    print(f"Number of classes: {cfg.model.backbone.num_classes}")
    print()
    
    # Build model
    print("Building model...")
    model = build_model(cfg.model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print("=" * 60)
    print("PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Total parameters:        {total_params:,}")
    print(f"                        {total_params/1e6:.2f}M")
    print(f"                        {total_params/1e9:.3f}B")
    print()
    print(f"Trainable parameters:    {trainable_params:,}")
    print(f"                        {trainable_params/1e6:.2f}M")
    print(f"                        {trainable_params/1e9:.3f}B")
    print()
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"                         {non_trainable_params/1e6:.2f}M")
    print()
    print(f"Trainable ratio:         {trainable_params/total_params*100:.2f}%")
    print("=" * 60)
    
    # Memory estimation (rough)
    param_memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"Estimated parameter memory: {param_memory_mb:.1f} MB")
    
    # Show layer-wise parameter counts
    print("\nLAYER-WISE PARAMETER BREAKDOWN:")
    print("-" * 60)
    
    def count_layer_params(module, name=""):
        """Recursively count parameters in each layer"""
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            print(f"{name:30}: {params:>10,} ({params/1e6:>6.2f}M)")
        
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            count_layer_params(child_module, full_name)
    
    count_layer_params(model)
    
    print("=" * 60)
    print("Model analysis completed!")
    print("=" * 60)

if __name__ == "__main__":
    config_path = '/home/vmuriki3/Documents/transformer/peachtree-pruning/PointTransformerV3/Pointcept/configs/scannet/semseg-pt-v2m2-0-peachtree.py'
    count_model_parameters(config_path)
