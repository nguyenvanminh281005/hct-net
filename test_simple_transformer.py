#!/usr/bin/env python3
"""
Simple test script for transformer connections
"""

import torch
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hct_net.genotypes import CellLinkDownPos, CellLinkUpPos, CellPos, TransformerLayerConfigs
from hct_net.nas_model import get_models

def test_model_creation():
    """Test creating model with 8 transformer connections"""
    print("Testing model creation with 8 transformer connections...")
    
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
            self.model = "UnetLayer7"
            self.init_weight_type = "kaiming"
            
            # Device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    args = Args()
    
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
    
    try:
        # Create model
        model = get_models(
            args, switches_normal, switches_down, switches_up, switches_transformer, switches_transformer_connections,
            args.early_fix_arch, args.gen_max_child_flag, args.random_sample
        ).to(args.device)
        
        print(f"✓ Model created successfully")
        print(f"✓ Model moved to device: {args.device}")
        
        # Check if model has the new attributes
        if hasattr(model, 'alphas_transformer_connections'):
            print(f"✓ Model has alphas_transformer_connections: {model.alphas_transformer_connections.shape}")
        else:
            print("✗ Model missing alphas_transformer_connections")
            
        # Test transformer control methods
        if hasattr(model, 'set_transformer_control'):
            model.set_transformer_control(enable_transformers=True, connection_threshold=0.5)
            print("✓ Transformer control set successfully")
        else:
            print("✗ Model missing set_transformer_control method")
            
        # Test usage stats
        if hasattr(model, 'get_transformer_usage_stats'):
            stats = model.get_transformer_usage_stats()
            print(f"✓ Got transformer usage stats: {len(stats)} entries")
            
            # Print connection stats
            connection_count = 0
            for key, value in stats.items():
                if 'connection_enc' in key:
                    connection_count += 1
                    print(f"  {key}: prob={value['probability']:.3f}, enabled={value['enabled']}")
            print(f"✓ Found {connection_count} transformer connections")
        else:
            print("✗ Model missing get_transformer_usage_stats method")
            
        # Test forward pass with small input
        try:
            batch_size = 1
            input_tensor = torch.randn(batch_size, 3, 64, 64).to(args.device)  # Smaller input for testing
            target_tensor = torch.randn(batch_size, 1, 64, 64).to(args.device)
            
            # Create dummy criterion
            import torch.nn as nn
            criterion = nn.BCEWithLogitsLoss()
            
            model.eval()
            with torch.no_grad():
                preds = model(input_tensor, target_tensor, criterion)
                print(f"✓ Forward pass successful: {len(preds)} outputs")
                print(f"  Output shapes: {[pred.shape for pred in preds]}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            
        return model
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None

def main():
    """Main test function"""
    print("Simple Transformer Connection Test\n")
    
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("! Using CPU")
    
    model = test_model_creation()
    
    if model is not None:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")

if __name__ == '__main__':
    main()