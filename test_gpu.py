#!/usr/bin/env python3
"""
Test script to check if GPU setup is working correctly
"""
import torch
import sys
import os

def main():
    print("Testing GPU setup...")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return False
    
    # GPU info
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Test basic tensor operations on GPU
    try:
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
        
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Perform computation
        z = torch.matmul(x, y)
        
        print("✓ Basic GPU operations successful")
        
        # Test model on GPU
        model = torch.nn.Linear(1000, 100).to(device)
        output = model(x)
        
        print("✓ Model operations on GPU successful")
        
        # Memory info
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"ERROR: GPU test failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n✓ GPU setup is working correctly!")
    else:
        print("\n✗ GPU setup has issues!")
        sys.exit(1)