"""
Visualize Chain Rule Gradient Flow for Transformer Connections

This script demonstrates how gradients flow through transformer connections
using the hierarchical chain rule approach.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def compute_parallel_gradient(probs_on, performance_score):
    """Original approach: Parallel gradient flow"""
    rewards = performance_score * probs_on
    return rewards.numpy()


def compute_chain_rule_gradient(probs_on, performance_score):
    """New approach: Hierarchical chain rule gradient flow"""
    num_connections = len(probs_on)
    cascading_rewards = []
    
    for i in range(num_connections):
        if i == 0:
            chain_factor = 1.0
        else:
            downstream_probs = probs_on[:i]
            chain_factor = torch.prod(downstream_probs + 0.1)
        
        connection_reward = performance_score * chain_factor * probs_on[i]
        cascading_rewards.append(connection_reward.item())
    
    return np.array(cascading_rewards)


def visualize_comparison():
    """Compare parallel vs chain rule gradient flow"""
    
    # Setup: 4 connections với xác suất ON khác nhau
    probs_on_scenarios = [
        torch.tensor([0.9, 0.7, 0.5, 0.3], dtype=torch.float32),  # Giảm dần
        torch.tensor([0.3, 0.5, 0.7, 0.9], dtype=torch.float32),  # Tăng dần
        torch.tensor([0.9, 0.9, 0.3, 0.3], dtype=torch.float32),  # Deep ON, shallow OFF
        torch.tensor([0.3, 0.3, 0.9, 0.9], dtype=torch.float32),  # Deep OFF, shallow ON
    ]
    
    scenario_names = [
        "Decreasing (0.9→0.3)",
        "Increasing (0.3→0.9)", 
        "Deep ON, Shallow OFF",
        "Deep OFF, Shallow ON"
    ]
    
    performance_score = torch.tensor(0.8)  # Good Dice performance
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (probs_on, name) in enumerate(zip(probs_on_scenarios, scenario_names)):
        ax = axes[idx]
        
        # Compute gradients
        parallel_rewards = compute_parallel_gradient(probs_on, performance_score)
        chain_rule_rewards = compute_chain_rule_gradient(probs_on, performance_score)
        
        x = np.arange(len(probs_on))
        width = 0.35
        
        # Plot bars
        bars1 = ax.bar(x - width/2, parallel_rewards, width, label='Parallel (Old)', 
                      color='#ff7f0e', alpha=0.8)
        bars2 = ax.bar(x + width/2, chain_rule_rewards, width, label='Chain Rule (New)', 
                      color='#2ca02c', alpha=0.8)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        # Add probs_on values below x-axis
        for i, prob in enumerate(probs_on.numpy()):
            ax.text(i, -0.05, f'p={prob:.1f}', 
                   ha='center', va='top', fontsize=8, color='blue')
        
        ax.set_xlabel('Connection Index (0=Deepest, 3=Shallowest)', fontsize=10)
        ax.set_ylabel('Reward (Gradient Strength)', fontsize=10)
        ax.set_title(f'Scenario: {name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Conn 0\n(Deep)', 'Conn 1', 'Conn 2', 'Conn 3\n(Shallow)'])
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(-0.1, 1.0)
    
    plt.tight_layout()
    plt.savefig('chain_rule_gradient_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: chain_rule_gradient_comparison.png")
    plt.close()


def visualize_gradient_flow_diagram():
    """Visualize the cascading gradient flow structure"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Setup
    num_connections = 4
    probs_on = torch.tensor([0.9, 0.7, 0.5, 0.3], dtype=torch.float32)
    performance_score = 0.8
    
    # Compute chain factors
    chain_factors = []
    for i in range(num_connections):
        if i == 0:
            chain_factors.append(1.0)
        else:
            downstream_probs = probs_on[:i]
            chain_factor = torch.prod(downstream_probs + 0.1).item()
            chain_factors.append(chain_factor)
    
    # Compute rewards
    rewards = [performance_score * chain_factors[i] * probs_on[i].item() 
               for i in range(num_connections)]
    
    # Draw flow diagram
    y_positions = np.arange(num_connections, 0, -1)  # Top to bottom
    x_center = 5
    
    # Draw Dice Loss at top
    ax.add_patch(plt.Rectangle((x_center - 1, y_positions[0] + 0.5), 2, 0.6, 
                               facecolor='red', edgecolor='black', linewidth=2))
    ax.text(x_center, y_positions[0] + 0.8, 'DICE LOSS', 
           ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Draw connections
    for i in range(num_connections):
        y = y_positions[i]
        
        # Connection box
        box_color = plt.cm.viridis(probs_on[i].item())
        ax.add_patch(plt.Rectangle((x_center - 1.5, y - 0.3), 3, 0.6, 
                                   facecolor=box_color, edgecolor='black', linewidth=2, alpha=0.8))
        
        # Connection label
        ax.text(x_center, y, f'Connection {i}\n(Layer {i+2}↔{9-i})', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Prob ON
        ax.text(x_center - 2.5, y, f'p_on = {probs_on[i].item():.2f}', 
               ha='right', va='center', fontsize=9, color='blue')
        
        # Chain factor
        ax.text(x_center + 2.5, y, f'chain = {chain_factors[i]:.3f}', 
               ha='left', va='center', fontsize=9, color='green')
        
        # Reward
        ax.text(x_center + 5, y, f'reward = {rewards[i]:.3f}', 
               ha='left', va='center', fontsize=9, fontweight='bold', color='red')
        
        # Arrow from previous
        if i == 0:
            # Arrow from Dice Loss
            ax.arrow(x_center, y_positions[0] + 0.5, 0, -0.3, 
                    head_width=0.3, head_length=0.1, fc='red', ec='red', linewidth=2)
            ax.text(x_center + 0.5, y_positions[0] + 0.3, 'Full gradient', 
                   fontsize=8, color='red', style='italic')
        else:
            # Arrow from previous connection
            arrow_color = plt.cm.viridis(chain_factors[i])
            ax.arrow(x_center, y_positions[i-1] - 0.3, 0, -0.5, 
                    head_width=0.3, head_length=0.1, fc=arrow_color, ec=arrow_color, 
                    linewidth=2, alpha=0.7)
            
            # Dependency text
            deps = ' × '.join([f'p_{j}' for j in range(i)])
            ax.text(x_center + 0.5, y + 0.5, f'gated by: {deps}', 
                   fontsize=7, color='purple', style='italic')
    
    ax.set_xlim(0, 11)
    ax.set_ylim(0, num_connections + 1.5)
    ax.axis('off')
    ax.set_title('Hierarchical Chain Rule Gradient Flow\n(Cascading from Output to Input)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_text = (
        "Key Insight:\n"
        "• Connection 0 (deepest): Full gradient from Dice Loss\n"
        "• Connection 1: Modulated by p_0 (depends on Connection 0)\n"
        "• Connection 2: Modulated by p_0 × p_1 (depends on 0 & 1)\n"
        "• Connection 3 (shallowest): Modulated by p_0 × p_1 × p_2\n\n"
        "→ Upstream layers depend on downstream decisions!"
    )
    ax.text(0.5, 0.3, legend_text, fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('chain_rule_gradient_flow_diagram.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: chain_rule_gradient_flow_diagram.png")
    plt.close()


def print_numerical_example():
    """Print detailed numerical example"""
    
    print("\n" + "="*70)
    print("NUMERICAL EXAMPLE: Chain Rule vs Parallel Gradient Flow")
    print("="*70)
    
    probs_on = torch.tensor([0.9, 0.7, 0.5, 0.3], dtype=torch.float32)
    performance_score = torch.tensor(0.8)
    
    print(f"\nSetup:")
    print(f"  Performance score (from Dice): {performance_score.item():.3f}")
    print(f"  Probs ON: {probs_on.numpy()}")
    print(f"  Connections: 0 (deepest, closest to output) → 3 (shallowest)")
    
    print(f"\n{'='*70}")
    print("PARALLEL APPROACH (Old):")
    print(f"{'='*70}")
    parallel_rewards = compute_parallel_gradient(probs_on, performance_score)
    for i, reward in enumerate(parallel_rewards):
        print(f"  Connection {i}: reward = {performance_score.item():.3f} × {probs_on[i].item():.3f} = {reward:.4f}")
    print(f"  Mean reward: {parallel_rewards.mean():.4f}")
    print(f"  Final loss: -5.0 × {parallel_rewards.mean():.4f} = {-5.0 * parallel_rewards.mean():.4f}")
    
    print(f"\n{'='*70}")
    print("CHAIN RULE APPROACH (New):")
    print(f"{'='*70}")
    
    chain_factors = []
    for i in range(len(probs_on)):
        if i == 0:
            chain_factor = 1.0
            print(f"  Connection {i} (deepest):")
            print(f"    Chain factor = 1.0 (direct from output)")
        else:
            downstream_probs = probs_on[:i]
            chain_factor = torch.prod(downstream_probs + 0.1).item()
            deps = ' × '.join([f'(p_{j}+0.1)' for j in range(i)])
            print(f"  Connection {i}:")
            print(f"    Chain factor = {deps}")
            print(f"                 = {chain_factor:.4f}")
        
        chain_factors.append(chain_factor)
        reward = performance_score.item() * chain_factor * probs_on[i].item()
        print(f"    Reward = {performance_score.item():.3f} × {chain_factor:.4f} × {probs_on[i].item():.3f} = {reward:.4f}")
        print()
    
    chain_rule_rewards = compute_chain_rule_gradient(probs_on, performance_score)
    print(f"  Mean reward: {chain_rule_rewards.mean():.4f}")
    print(f"  Final loss: -5.0 × {chain_rule_rewards.mean():.4f} = {-5.0 * chain_rule_rewards.mean():.4f}")
    
    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    print(f"{'Connection':<15} {'Parallel':<12} {'Chain Rule':<12} {'Difference':<12}")
    print(f"{'-'*70}")
    for i in range(len(probs_on)):
        diff = chain_rule_rewards[i] - parallel_rewards[i]
        sign = "↑" if diff > 0 else "↓"
        print(f"Conn {i} (L{i+2}↔L{9-i}): {parallel_rewards[i]:>10.4f}  {chain_rule_rewards[i]:>10.4f}  {diff:>10.4f} {sign}")
    
    print(f"\nKey Observations:")
    print(f"  • Connection 0 (deepest): Chain rule ≈ Parallel (full gradient)")
    print(f"  • Connection 1-3: Chain rule < Parallel (modulated by upstream)")
    print(f"  • Gradient strength: Deep > Shallow (hierarchical learning)")
    print(f"  • Deep connections optimized first, shallow connections follow")


if __name__ == '__main__':
    print("Generating Chain Rule Gradient Flow Visualizations...")
    print()
    
    # Generate plots
    visualize_comparison()
    visualize_gradient_flow_diagram()
    
    # Print numerical example
    print_numerical_example()
    
    print(f"\n{'='*70}")
    print("✅ DONE! Generated 2 visualization files:")
    print("   1. chain_rule_gradient_comparison.png")
    print("   2. chain_rule_gradient_flow_diagram.png")
    print(f"{'='*70}\n")
