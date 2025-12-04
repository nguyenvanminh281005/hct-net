#!/usr/bin/env python3
"""
Test script for complexity loss and transformer control features
"""

import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hct_net.genotypes import CellLinkDownPos, CellLinkUpPos, CellPos, TransformerLayerConfigs
from hct_net.nas_model import get_models

def create_test_args():
    """Create test arguments"""
    class Args:
        def __init__(self):
            # Model parameters
            self.input_c = 3
            self.init_channel = 16
            self.num_classes = 1
            self.meta_node_num = 4
            self.layers = 7
            self.use_sharing = True
            self.depth = 4
            self.double_down_channel = True
            self.dropout_prob = 0
            self.use_softmax_head = False
            self.early_fix_arch = True
            self.gen_max_child = True
            self.gen_max_child_flag = False
            self.random_sample = False
            
            # Device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Complexity loss weights
            self.complexity_weight = 0.01
            self.arch_complexity_weight = 0.005
            self.transformer_connection_weight = 0.02
            
            # Transformer control
            self.enable_transformers = True
            self.transformer_threshold = 2
            self.adaptive_transformer_control = False
    
    return Args()

def test_complexity_functions():
    """Test complexity loss functions"""
    print("=== Testing Complexity Loss Functions ===")
    
    args = create_test_args()
    
    # Import functions
    from hct_net.train_CVCDataset import (
        compute_complexity_loss, 
        compute_architecture_complexity_loss,
        compute_transformer_connection_loss
    )
    
    # Create switches
    normal_num_ops = len(CellPos)
    down_num_ops = len(CellLinkDownPos) 
    up_num_ops = len(CellLinkUpPos)
    transformer_num_configs = len(TransformerLayerConfigs)
    
    switches_normal = []
    switches_down = []
    switches_up = []
    switches_transformer = []
    switches_transformer_connections = []
    
    for i in range(14):  # 14 edges in the search space
        switches_normal.append([True] * normal_num_ops)
        switches_down.append([True] * down_num_ops)  
        switches_up.append([True] * up_num_ops)
        
    for i in range(args.layers):  # transformer switches for each layer
        switches_transformer.append([True] * transformer_num_configs)
        
    # 8 transformer connections for UNet 7 layers
    for i in range(8):
        switches_transformer_connections.append([True])
    
    # Create model
    model = get_models(
        args, switches_normal, switches_down, switches_up, switches_transformer, switches_transformer_connections,
        args.early_fix_arch, args.gen_max_child_flag, args.random_sample
    ).to(args.device)
    
    # Test complexity loss
    try:
        complexity_loss = compute_complexity_loss(model, args)
        print(f"✓ Complexity loss: {complexity_loss.item():.4f}")
    except Exception as e:
        print(f"✗ Complexity loss failed: {e}")
    
    # Test architecture complexity loss
    try:
        arch_complexity_loss = compute_architecture_complexity_loss(model, args)
        print(f"✓ Architecture complexity loss: {arch_complexity_loss.item():.4f}")
    except Exception as e:
        print(f"✗ Architecture complexity loss failed: {e}")
    
    # Test transformer connection loss
    try:
        transformer_loss = compute_transformer_connection_loss(model, args)
        print(f"✓ Transformer connection loss: {transformer_loss.item():.4f}")
    except Exception as e:
        print(f"✗ Transformer connection loss failed: {e}")
    
    return model

def test_transformer_control(model):
    """Test transformer connection control"""
    print("\n=== Testing Transformer Control ===")
    
    if not hasattr(model, 'set_transformer_control'):
        print("✗ Model does not have transformer control methods")
        return
    
    # Test enabling transformers
    try:
        model.set_transformer_control(enable_transformers=True, transformer_threshold=2)
        print("✓ Transformer control enabled")
    except Exception as e:
        print(f"✗ Failed to enable transformer control: {e}")
    
    # Test disabling transformers  
    try:
        model.set_transformer_control(enable_transformers=False)
        print("✓ Transformer control disabled")
    except Exception as e:
        print(f"✗ Failed to disable transformer control: {e}")
    
    # Test getting usage stats
    try:
        if hasattr(model, 'get_transformer_usage_stats'):
            stats = model.get_transformer_usage_stats()
            if stats:
                print(f"✓ Transformer usage stats: {len(stats)} layers")
                for layer, stat in stats.items():
                    print(f"  {layer}: {stat['num_layers']} layers, enabled={stat['enabled']}")
            else:
                print("! No transformer usage stats available")
        else:
            print("! Model does not have get_transformer_usage_stats method")
    except Exception as e:
        print(f"✗ Failed to get transformer usage stats: {e}")

def test_model_forward():
    """Test model forward pass"""
    print("\n=== Testing Model Forward Pass ===")
    
    args = create_test_args()
    
    # Create switches
    normal_num_ops = len(CellPos)
    down_num_ops = len(CellLinkDownPos)
    up_num_ops = len(CellLinkUpPos)
    transformer_num_configs = len(TransformerLayerConfigs)
    
    switches_normal = []
    switches_down = []
    switches_up = []
    switches_transformer = []
    switches_transformer_connections = []
    
    for i in range(14):
        switches_normal.append([True] * normal_num_ops)
        switches_down.append([True] * down_num_ops)
        switches_up.append([True] * up_num_ops)
        
    for i in range(args.layers):
        switches_transformer.append([True] * transformer_num_configs)
        
    # 8 transformer connections for UNet 7 layers
    for i in range(8):
        switches_transformer_connections.append([True])
    
    # Create model and move to GPU
    model = get_models(
        args, switches_normal, switches_down, switches_up, switches_transformer, switches_transformer_connections,
        args.early_fix_arch, args.gen_max_child_flag, args.random_sample
    ).to(args.device)
    
    # Create dummy input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256).to(args.device)
    target_tensor = torch.randn(batch_size, 1, 256, 256).to(args.device)
    
    # Create dummy criterion
    criterion = nn.BCEWithLogitsLoss()
    
    # Test forward pass with transformers enabled
    try:
        model.set_transformer_control(enable_transformers=True, transformer_threshold=1)
        model.train()
        preds = model(input_tensor, target_tensor, criterion)
        print(f"✓ Forward pass with transformers enabled: {len(preds)} outputs")
        print(f"  Output shapes: {[pred.shape for pred in preds]}")
    except Exception as e:
        print(f"✗ Forward pass with transformers failed: {e}")
    
    # Test forward pass with transformers disabled
    try:
        model.set_transformer_control(enable_transformers=False)
        preds = model(input_tensor, target_tensor, criterion)
        print(f"✓ Forward pass with transformers disabled: {len(preds)} outputs")
        print(f"  Output shapes: {[pred.shape for pred in preds]}")
    except Exception as e:
        print(f"✗ Forward pass without transformers failed: {e}")

def main():
    """Main test function"""
    print("Testing Complexity Loss and Transformer Control Features\n")
    
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("! Using CPU")
    
    # Test complexity functions
    model = test_complexity_functions()
    
    # Test transformer control
    test_transformer_control(model)
    
    # Test model forward pass
    test_model_forward()
    
    print("\n=== Test Summary ===")
    print("All tests completed!")
    print("Check the output above for any failed tests (marked with ✗)")

if __name__ == '__main__':
    main()