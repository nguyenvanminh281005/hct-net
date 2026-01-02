"""
Script để test và visualize Pareto V2 multi-objective optimization

Chạy script này để:
1. Test complexity calculation
2. Visualize Pareto frontier
3. Demo transformer configuration search space
"""

import sys
sys.path.append('./hct_net')

import torch
import torch.nn.functional as F
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False
    
from hct_net.train_CVCDataset_pareto_v2 import (
    TRANSFORMER_CONFIG_CHOICES,
    NUM_TRANSFORMER_CONFIGS,
    TRANSFORMER_COMPLEXITY_LOOKUP,
    calculate_transformer_complexity,
    compute_pareto_loss,
)

print("="*80)
print("PARETO V2 MULTI-OBJECTIVE OPTIMIZATION - DEMO")
print("="*80)

# ================================================================================
# 1. Show transformer configuration search space
# ================================================================================

print("\n1. TRANSFORMER CONFIGURATION SEARCH SPACE")
print("-"*80)
print(f"Total configurations: {NUM_TRANSFORMER_CONFIGS}")
print(f"\nSample configurations (first 10):")

for i in range(min(10, NUM_TRANSFORMER_CONFIGS)):
    config = TRANSFORMER_CONFIG_CHOICES[i]
    complexity = TRANSFORMER_COMPLEXITY_LOOKUP[i]
    print(f"  Config {i:2d}: d_model={config['d_model']:3d}, n_head={config['n_head']}, "
          f"expansion={config['expansion']} → "
          f"FLOPs={complexity['flops']:6.2f}M, Params={complexity['params']:5.2f}M, "
          f"Score={complexity['complexity_score']:6.2f}")

# ================================================================================
# 2. Complexity comparison
# ================================================================================

print("\n2. COMPLEXITY COMPARISON")
print("-"*80)

# Find min, mid, max complexity configs
complexities = [(i, TRANSFORMER_COMPLEXITY_LOOKUP[i]['complexity_score']) 
                for i in range(NUM_TRANSFORMER_CONFIGS)]
complexities.sort(key=lambda x: x[1])

min_idx, min_complexity = complexities[0]
mid_idx, mid_complexity = complexities[len(complexities)//2]
max_idx, max_complexity = complexities[-1]

print(f"\nMinimum complexity configuration (Index {min_idx}):")
config = TRANSFORMER_CONFIG_CHOICES[min_idx]
print(f"  d_model={config['d_model']}, n_head={config['n_head']}, expansion={config['expansion']}")
print(f"  FLOPs={TRANSFORMER_COMPLEXITY_LOOKUP[min_idx]['flops']:.2f}M, "
      f"Params={TRANSFORMER_COMPLEXITY_LOOKUP[min_idx]['params']:.2f}M")

print(f"\nMedium complexity configuration (Index {mid_idx}):")
config = TRANSFORMER_CONFIG_CHOICES[mid_idx]
print(f"  d_model={config['d_model']}, n_head={config['n_head']}, expansion={config['expansion']}")
print(f"  FLOPs={TRANSFORMER_COMPLEXITY_LOOKUP[mid_idx]['flops']:.2f}M, "
      f"Params={TRANSFORMER_COMPLEXITY_LOOKUP[mid_idx]['params']:.2f}M")

print(f"\nMaximum complexity configuration (Index {max_idx}):")
config = TRANSFORMER_CONFIG_CHOICES[max_idx]
print(f"  d_model={config['d_model']}, n_head={config['n_head']}, expansion={config['expansion']}")
print(f"  FLOPs={TRANSFORMER_COMPLEXITY_LOOKUP[max_idx]['flops']:.2f}M, "
      f"Params={TRANSFORMER_COMPLEXITY_LOOKUP[max_idx]['params']:.2f}M")

print(f"\nComplexity ratio (max/min): {max_complexity/min_complexity:.1f}x")

# ================================================================================
# 3. Simulate Pareto optimization
# ================================================================================

print("\n3. PARETO OPTIMIZATION SIMULATION")
print("-"*80)

class DummyArgs:
    device = 'cpu'
    pareto_weight_dice = 1.0
    pareto_weight_complexity = 5.0  # HIGH WEIGHT
    pareto_weight_connection = 2.0

args = DummyArgs()

# Simulate different scenarios
scenarios = [
    {
        'name': 'High accuracy, high complexity',
        'dice_loss': 0.1,  # Good accuracy
        'complexity': 100.0,  # High complexity (100M FLOPs)
        'connection_var': 0.05,  # Low variance (all similar)
    },
    {
        'name': 'Medium accuracy, low complexity',
        'dice_loss': 0.3,  # Medium accuracy
        'complexity': 20.0,  # Low complexity
        'connection_var': 0.2,  # High variance (differentiated)
    },
    {
        'name': 'Low accuracy, very low complexity',
        'dice_loss': 0.5,  # Poor accuracy
        'complexity': 5.0,  # Very low complexity
        'connection_var': 0.3,  # Very high variance
    },
    {
        'name': 'Balanced solution',
        'dice_loss': 0.2,  # Good accuracy
        'complexity': 30.0,  # Moderate complexity
        'connection_var': 0.25,  # Good variance
    },
]

print("\nScenario comparison (lower Pareto loss is better):")
print(f"  Weights: Dice={args.pareto_weight_dice}, "
      f"Complexity={args.pareto_weight_complexity}, "
      f"Connection={args.pareto_weight_connection}")
print()

for scenario in scenarios:
    dice_loss = torch.tensor(scenario['dice_loss'])
    complexity_loss = torch.tensor(scenario['complexity'])
    connection_loss = torch.tensor(scenario['connection_var'])
    
    pareto_loss = compute_pareto_loss(dice_loss, complexity_loss, connection_loss, args)
    
    print(f"  {scenario['name']}")
    print(f"    Dice: {dice_loss:.3f}, Complexity: {complexity_loss:.1f}M, "
          f"Connection: {connection_loss:.3f}")
    print(f"    → Pareto Loss: {pareto_loss:.4f}")
    print()

# ================================================================================
# 4. Visualize Pareto frontier
# ================================================================================

print("\n4. PARETO FRONTIER VISUALIZATION")
print("-"*80)

# Generate random architecture samples
np.random.seed(42)
num_samples = 100

dice_losses = np.random.uniform(0.1, 0.6, num_samples)  # 0.1-0.6 Dice loss
complexities = np.random.uniform(5, 150, num_samples)  # 5-150M FLOPs

# Calculate Pareto loss for each sample
pareto_losses = []
for dice, comp in zip(dice_losses, complexities):
    connection_loss_val = np.random.uniform(0.05, 0.3)  # Random connection variance
    dice_t = torch.tensor(dice)
    comp_t = torch.tensor(comp)
    conn_t = torch.tensor(connection_loss_val)
    pareto_loss = compute_pareto_loss(dice_t, comp_t, conn_t, args)
    pareto_losses.append(pareto_loss.item())

pareto_losses = np.array(pareto_losses)

# Find Pareto frontier (non-dominated solutions)
def is_dominated(idx, dice_losses, complexities):
    """Check if solution at idx is dominated by any other solution"""
    for i in range(len(dice_losses)):
        if i == idx:
            continue
        # A solution dominates if it's better in both objectives
        if dice_losses[i] <= dice_losses[idx] and complexities[i] <= complexities[idx]:
            if dice_losses[i] < dice_losses[idx] or complexities[i] < complexities[idx]:
                return True
    return False

pareto_frontier_mask = np.array([not is_dominated(i, dice_losses, complexities) 
                                 for i in range(num_samples)])

print(f"Generated {num_samples} random architecture samples")
print(f"Found {pareto_frontier_mask.sum()} solutions on Pareto frontier")
print(f"\nPareto frontier solutions:")
frontier_indices = np.where(pareto_frontier_mask)[0]
for idx in frontier_indices[:5]:  # Show first 5
    print(f"  Dice: {dice_losses[idx]:.3f}, Complexity: {complexities[idx]:.1f}M, "
          f"Pareto Loss: {pareto_losses[idx]:.4f}")

# Create visualization
if HAS_MATPLOTLIB:
  try:
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Dice vs Complexity with Pareto frontier
    plt.subplot(1, 2, 1)
    plt.scatter(dice_losses[~pareto_frontier_mask], complexities[~pareto_frontier_mask], 
               c='lightblue', alpha=0.5, label='Dominated solutions')
    plt.scatter(dice_losses[pareto_frontier_mask], complexities[pareto_frontier_mask], 
               c='red', s=100, marker='*', label='Pareto frontier', zorder=5)
    plt.xlabel('Dice Loss (lower is better)')
    plt.ylabel('Complexity (FLOPs in millions)')
    plt.title('Pareto Frontier: Dice vs Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Pareto loss distribution
    plt.subplot(1, 2, 2)
    plt.hist(pareto_losses, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(pareto_losses[pareto_frontier_mask].min(), color='red', 
                linestyle='--', linewidth=2, label='Min Pareto loss (frontier)')
    plt.xlabel('Pareto Loss')
    plt.ylabel('Frequency')
    plt.title('Distribution of Pareto Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pareto_v2_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: pareto_v2_demo.png")
    
  except Exception as e:
    print(f"\n⚠ Could not create visualization: {e}")
else:
    print(f"\n⚠ Matplotlib not available, skipping visualization")

# ================================================================================
# 5. Simulate transformer configuration selection
# ================================================================================

print("\n5. TRANSFORMER CONFIGURATION SELECTION SIMULATION")
print("-"*80)

# Simulate alphas for 4 transformer connections
num_connections = 4
alphas_configs = torch.randn(num_connections, NUM_TRANSFORMER_CONFIGS) * 0.1

# Add bias toward different complexity levels for each connection
# Connection 0 (deepest): bias toward large configs
alphas_configs[0, -5:] += 0.5  # Boost last 5 (large) configs

# Connection 1: bias toward medium configs
mid_start = NUM_TRANSFORMER_CONFIGS // 2 - 3
mid_end = NUM_TRANSFORMER_CONFIGS // 2 + 3
alphas_configs[1, mid_start:mid_end] += 0.5

# Connection 2: bias toward small configs
alphas_configs[2, :5] += 0.5

# Connection 3: uniform (will be off)
# Keep random

# Simulate connection on/off
alphas_connections = torch.tensor([
    [-0.5, 1.0],  # Connection 0: strongly ON
    [-0.2, 0.5],  # Connection 1: moderately ON
    [0.3, 0.1],   # Connection 2: moderately OFF
    [1.0, -0.5],  # Connection 3: strongly OFF
])

config_probs = F.softmax(alphas_configs, dim=-1)
conn_probs = F.softmax(alphas_connections, dim=-1)

print("Simulated architecture after search:")
print()

total_flops = 0.0
total_params = 0.0

for conn_idx in range(num_connections):
    selected_config_idx = torch.argmax(config_probs[conn_idx]).item()
    config = TRANSFORMER_CONFIG_CHOICES[selected_config_idx]
    complexity = TRANSFORMER_COMPLEXITY_LOOKUP[selected_config_idx]
    
    prob_on = conn_probs[conn_idx, 1].item()
    state = "ON" if prob_on > 0.5 else "OFF"
    
    print(f"Connection {conn_idx} [{state}, prob={prob_on:.3f}]:")
    print(f"  Selected Config {selected_config_idx}: "
          f"d_model={config['d_model']}, n_head={config['n_head']}, expansion={config['expansion']}")
    print(f"  Complexity: FLOPs={complexity['flops']:.2f}M, Params={complexity['params']:.2f}M")
    
    if prob_on > 0.5:
        total_flops += complexity['flops']
        total_params += complexity['params']
    
    print()

active_count = sum(1 for i in range(num_connections) if conn_probs[i, 1] > 0.5)
print(f"Total: {active_count}/{num_connections} transformers active")
print(f"Total complexity: FLOPs={total_flops:.2f}M, Params={total_params:.2f}M")

# ================================================================================
# 6. Connection differentiation analysis
# ================================================================================

print("\n6. CONNECTION DIFFERENTIATION ANALYSIS")
print("-"*80)

# Compare different connection probability distributions
distributions = [
    {
        'name': 'All ON (bad)',
        'probs': torch.tensor([0.95, 0.95, 0.95, 0.95]),
    },
    {
        'name': 'All OFF (bad)',
        'probs': torch.tensor([0.05, 0.05, 0.05, 0.05]),
    },
    {
        'name': 'All similar (bad)',
        'probs': torch.tensor([0.55, 0.52, 0.48, 0.50]),
    },
    {
        'name': 'Well differentiated (good)',
        'probs': torch.tensor([0.95, 0.75, 0.25, 0.05]),
    },
    {
        'name': 'Balanced diversity (good)',
        'probs': torch.tensor([0.90, 0.60, 0.40, 0.10]),
    },
]

print("Analyzing connection probability distributions:")
print("(Lower differentiation loss is better)")
print()

for dist in distributions:
    probs = dist['probs']
    
    # Calculate metrics
    mean_prob = probs.mean().item()
    variance = torch.var(probs).item()
    
    # Differentiation loss components
    target_ratio = 0.5
    ratio_penalty = (mean_prob - target_ratio) ** 2
    variance_penalty = -variance  # Negative because we maximize variance
    diff_loss = ratio_penalty + 0.5 * variance_penalty
    
    print(f"{dist['name']}:")
    print(f"  Probabilities: {[f'{p:.2f}' for p in probs.tolist()]}")
    print(f"  Mean: {mean_prob:.3f}, Variance: {variance:.3f}")
    print(f"  Ratio penalty: {ratio_penalty:.4f}, Variance penalty: {variance_penalty:.4f}")
    print(f"  → Differentiation loss: {diff_loss:.4f}")
    print()

print("="*80)
print("DEMO COMPLETED")
print("="*80)
print("\nKey takeaways:")
print("1. Search space has 30+ transformer configurations with vastly different complexities")
print("2. Pareto optimization balances accuracy, complexity, and connection differentiation")
print("3. High complexity weight (5.0) encourages selecting smaller, diverse configurations")
print("4. Well-differentiated connections have high variance and balanced ON/OFF ratio")
print("5. The system can automatically discover efficient architectures on the Pareto frontier")
