"""
Test script to verify transformer learning logic
"""
import torch
import torch.nn.functional as F
import numpy as np

def test_initialization_bias():
    """Test different initialization bias values"""
    print("="*80)
    print("TEST 1: Initialization Bias Impact")
    print("="*80)
    
    num_connections = 4
    bias_values = [0.0, 0.3, 0.5, 1.0, 2.0]
    
    for bias in bias_values:
        alphas = torch.zeros(num_connections, 2)
        alphas[:, 0] = -bias  # OFF
        alphas[:, 1] = bias   # ON
        
        probs = F.softmax(alphas, dim=-1)
        on_probs = probs[:, 1]
        
        print(f"\nBias={bias:.1f}:")
        print(f"  Initial ON probabilities: {[f'{p:.3f}' for p in on_probs.tolist()]}")
        print(f"  Average ON prob: {on_probs.mean():.3f}")
        print(f"  Expected ON count: {sum(1 for p in on_probs if p > 0.5)}/{num_connections}")
        print(f"  Exploration level: {'HIGH' if bias < 0.5 else 'MEDIUM' if bias < 1.0 else 'LOW'}")


def test_fixing_logic():
    """Test transformer fixing logic with different margins"""
    print("\n" + "="*80)
    print("TEST 2: Fixing Logic with Different Margins")
    print("="*80)
    
    # Simulate probabilities after some training
    prob_scenarios = [
        ("Balanced", [0.49, 0.51, 0.48, 0.52]),
        ("Moderate", [0.30, 0.70, 0.40, 0.60]),
        ("Strong", [0.10, 0.90, 0.20, 0.80]),
        ("Very Strong", [0.05, 0.95, 0.02, 0.98])
    ]
    
    margins = [0.3, 0.5, 0.6]
    
    for scenario_name, on_probs in prob_scenarios:
        print(f"\n{scenario_name} scenario - ON probs: {on_probs}")
        
        for margin in margins:
            # Convert to alphas (inverse softmax approximation)
            alphas = torch.zeros(len(on_probs), 2)
            for i, p in enumerate(on_probs):
                if p > 0.5:
                    # ON is more likely
                    diff = p - (1-p)
                    alphas[i, 1] = diff * 2
                    alphas[i, 0] = -diff * 2
                else:
                    # OFF is more likely
                    diff = (1-p) - p
                    alphas[i, 0] = diff * 2
                    alphas[i, 1] = -diff * 2
            
            probs = F.softmax(alphas, dim=-1)
            sort_probs = torch.topk(probs, 2, dim=-1)
            
            # Check which would be fixed
            fixed = (sort_probs.values[:, 0] - sort_probs.values[:, 1] >= margin)
            fixed_count = fixed.sum().item()
            
            print(f"  Margin={margin}: Would fix {fixed_count}/{len(on_probs)} connections")


def test_transformer_loss_gradient():
    """Test that transformer loss has gradient from accuracy"""
    print("\n" + "="*80)
    print("TEST 3: Transformer Loss Gradient Flow")
    print("="*80)
    
    # Simulate transformer alphas
    alphas_transformer = torch.tensor([
        [-0.3, 0.3],  # ~57% ON
        [-0.3, 0.3],
        [-0.3, 0.3],
        [-0.3, 0.3]
    ], requires_grad=True)
    
    # Simulate accuracy loss (should have gradient)
    accuracy_loss = torch.tensor(0.5, requires_grad=True)
    
    # Compute transformer loss (simplified version)
    probs = F.softmax(alphas_transformer, dim=-1)
    probs_on = probs[:, 1]
    
    # Performance-based loss
    performance_score = 1.0 - torch.sigmoid(accuracy_loss - 0.4)  # Relative to baseline
    transformer_loss = -(performance_score * probs_on).mean()
    
    # Backward
    transformer_loss.backward()
    
    print(f"Accuracy loss: {accuracy_loss.item():.4f}")
    print(f"Performance score: {performance_score.item():.4f}")
    print(f"Transformer loss: {transformer_loss.item():.4f}")
    print(f"Alphas grad: {alphas_transformer.grad}")
    print(f"Gradient norm: {alphas_transformer.grad.norm().item():.6f}")
    
    has_grad = alphas_transformer.grad is not None and alphas_transformer.grad.abs().max() > 1e-6
    print(f"\nGradient flow: {'✓ YES' if has_grad else '✗ NO'}")
    
    if has_grad:
        # Check which direction gradients push
        grad_on = alphas_transformer.grad[:, 1].mean().item()
        if grad_on > 0:
            print(f"Gradient direction: ENCOURAGES ON (good performance)")
        else:
            print(f"Gradient direction: DISCOURAGES ON (bad performance)")


def test_warmup_schedule():
    """Test warmup scheduling"""
    print("\n" + "="*80)
    print("TEST 4: Warmup Schedule")
    print("="*80)
    
    warmup_epochs = 5
    total_epochs = 20
    
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Total epochs: {total_epochs}")
    print("\nWarmup bonus schedule:")
    
    for epoch in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
        if epoch < warmup_epochs:
            warmup_progress = 1.0 - (epoch / warmup_epochs)
            warmup_bonus = -warmup_progress * 1.0  # Negative = reward
            status = "WARMUP"
        else:
            warmup_bonus = 0.0
            status = "NORMAL"
        
        print(f"  Epoch {epoch:2d}: bonus={warmup_bonus:+.3f} [{status}]")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRANSFORMER LEARNING VERIFICATION TESTS")
    print("="*80 + "\n")
    
    test_initialization_bias()
    test_fixing_logic()
    test_transformer_loss_gradient()
    test_warmup_schedule()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80 + "\n")
    
    print("KEY RECOMMENDATIONS:")
    print("1. Use transformer_init_bias=0.3 for balanced exploration")
    print("2. Set fix_transformer_arch=False to disable early fixing")
    print("3. Use transformer_fix_margin=0.6 if fixing is enabled")
    print("4. Set transformer_min_epochs_before_fix >= warmup_epochs + 10")
    print("5. Monitor TransGrad to verify gradient flow")
