"""
Phân tích kết quả Ablation Study
Tự động thu thập và so sánh kết quả từ 8 cấu hình ablation
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import numpy as np

# Thiết lập style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def collect_ablation_results(search_exp_dir='./search_exp//Nas_Search_Unet/cvc'):
    """
    Thu thập kết quả từ tất cả các thí nghiệm ablation.
    
    Args:
        search_exp_dir: Thư mục chứa kết quả experiments
        
    Returns:
        DataFrame với kết quả từ 8 cấu hình
    """
    results = []
    
    # 8 modes cần tìm
    ablation_modes = ['all', 'no_transformer', 'no_complexity', 'no_entropy',
                      'only_transformer', 'only_complexity', 'only_entropy', 'none']
    
    search_path = Path(search_exp_dir)
    if not search_path.exists():
        print(f"Warning: Directory not found: {search_exp_dir}")
        return None
    
    # Tìm tất cả các thư mục ablation
    for mode in ablation_modes:
        mode_dirs = list(search_path.glob(f'ablation_{mode}_*'))
        
        if not mode_dirs:
            print(f"Warning: No results found for mode '{mode}'")
            continue
        
        # Lấy thư mục mới nhất
        latest_dir = max(mode_dirs, key=lambda p: p.stat().st_mtime)
        
        # Đọc checkpoint
        checkpoint_path = latest_dir / 'single' / 'checkpoint.pth.tar'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            result = {
                'mode': mode,
                'dice': checkpoint.get('dice', 0),
                'jaccard': checkpoint.get('jc', 0),
                'accuracy': checkpoint.get('accuracy', 0),
                'epoch': checkpoint.get('epoch', 0),
                'best_dice': checkpoint.get('best_dice', checkpoint.get('dice', 0)),
                'path': str(latest_dir),
            }
            results.append(result)
            print(f"✓ Loaded {mode}: Dice={result['dice']:.4f}, Jaccard={result['jaccard']:.4f}")
        else:
            print(f"Warning: Checkpoint not found for mode '{mode}' at {checkpoint_path}")
    
    if not results:
        print("No results found!")
        return None
    
    df = pd.DataFrame(results)
    
    # Thêm component flags
    df['has_transformer'] = ~df['mode'].str.contains('no_transformer|only_complexity|only_entropy|none')
    df['has_complexity'] = ~df['mode'].str.contains('no_complexity|only_transformer|only_entropy|none')
    df['has_entropy'] = ~df['mode'].str.contains('no_entropy|only_transformer|only_complexity|none')
    
    return df


def plot_ablation_comparison(df, output_dir='./ablation_analysis'):
    """
    Vẽ biểu đồ so sánh kết quả ablation.
    
    Args:
        df: DataFrame với kết quả
        output_dir: Thư mục lưu biểu đồ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. So sánh Dice Score
    plt.figure(figsize=(14, 6))
    
    # Sắp xếp theo Dice score
    df_sorted = df.sort_values('dice', ascending=False)
    
    colors = ['#2ecc71' if mode == 'all' else '#e74c3c' if mode == 'none' else '#3498db' 
              for mode in df_sorted['mode']]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(df_sorted)), df_sorted['dice'], color=colors)
    plt.xticks(range(len(df_sorted)), df_sorted['mode'], rotation=45, ha='right')
    plt.ylabel('Dice Score')
    plt.title('Ablation Study: Dice Score Comparison')
    plt.axhline(y=df_sorted[df_sorted['mode'] == 'none']['dice'].values[0] if 'none' in df_sorted['mode'].values else 0, 
                color='red', linestyle='--', label='Baseline (none)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Thêm giá trị lên bar
    for i, (bar, val) in enumerate(zip(bars, df_sorted['dice'])):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 2. So sánh Jaccard Index
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(len(df_sorted)), df_sorted['jaccard'], color=colors)
    plt.xticks(range(len(df_sorted)), df_sorted['mode'], rotation=45, ha='right')
    plt.ylabel('Jaccard Index')
    plt.title('Ablation Study: Jaccard Index Comparison')
    plt.axhline(y=df_sorted[df_sorted['mode'] == 'none']['jaccard'].values[0] if 'none' in df_sorted['mode'].values else 0,
                color='red', linestyle='--', label='Baseline (none)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, df_sorted['jaccard'])):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/ablation_metrics_comparison.png")
    
    # 3. Heatmap: Component contribution
    plt.figure(figsize=(10, 6))
    
    # Tạo matrix cho heatmap
    component_matrix = df[['has_transformer', 'has_complexity', 'has_entropy', 'dice']].copy()
    component_matrix['config'] = df['mode']
    
    pivot = component_matrix.pivot_table(
        index=['has_transformer', 'has_complexity', 'has_entropy'],
        values='dice',
        aggfunc='mean'
    )
    
    plt.subplot(1, 2, 1)
    contribution_data = {
        'Component': ['Baseline (Dice only)', '+ Transformer', '+ Complexity', '+ Entropy', 'Full (All)'],
        'Dice Score': [
            df[df['mode'] == 'none']['dice'].values[0] if 'none' in df['mode'].values else 0,
            df[df['mode'] == 'only_transformer']['dice'].values[0] if 'only_transformer' in df['mode'].values else 0,
            df[df['mode'] == 'only_complexity']['dice'].values[0] if 'only_complexity' in df['mode'].values else 0,
            df[df['mode'] == 'only_entropy']['dice'].values[0] if 'only_entropy' in df['mode'].values else 0,
            df[df['mode'] == 'all']['dice'].values[0] if 'all' in df['mode'].values else 0,
        ]
    }
    contrib_df = pd.DataFrame(contribution_data)
    
    colors_contrib = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#2ecc71']
    bars = plt.bar(contrib_df['Component'], contrib_df['Dice Score'], color=colors_contrib)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Dice Score')
    plt.title('Individual Component Contribution')
    plt.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, contrib_df['Dice Score']):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 4. Delta from baseline
    plt.subplot(1, 2, 2)
    baseline_dice = df[df['mode'] == 'none']['dice'].values[0] if 'none' in df['mode'].values else 0
    df_sorted['delta'] = df_sorted['dice'] - baseline_dice
    
    colors_delta = ['green' if x > 0 else 'red' for x in df_sorted['delta']]
    bars = plt.bar(range(len(df_sorted)), df_sorted['delta'], color=colors_delta, alpha=0.7)
    plt.xticks(range(len(df_sorted)), df_sorted['mode'], rotation=45, ha='right')
    plt.ylabel('Δ Dice Score (vs Baseline)')
    plt.title('Improvement over Baseline')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, df_sorted['delta']):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.002 if val > 0 else val - 0.002, 
                f'{val:+.3f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_contribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/ablation_contribution_analysis.png")
    
    plt.close('all')


def generate_summary_table(df, output_dir='./ablation_analysis'):
    """
    Tạo bảng tổng kết kết quả.
    
    Args:
        df: DataFrame với kết quả
        output_dir: Thư mục lưu bảng
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tính baseline
    baseline_dice = df[df['mode'] == 'none']['dice'].values[0] if 'none' in df['mode'].values else 0
    baseline_jaccard = df[df['mode'] == 'none']['jaccard'].values[0] if 'none' in df['mode'].values else 0
    
    summary = df.copy()
    summary['dice_delta'] = summary['dice'] - baseline_dice
    summary['jaccard_delta'] = summary['jaccard'] - baseline_jaccard
    
    # Sắp xếp theo Dice score
    summary = summary.sort_values('dice', ascending=False)
    
    # Chỉ giữ các cột quan trọng
    summary_table = summary[['mode', 'dice', 'dice_delta', 'jaccard', 'jaccard_delta', 
                             'has_transformer', 'has_complexity', 'has_entropy']]
    
    # Rename columns
    summary_table.columns = ['Configuration', 'Dice', 'Δ Dice', 'Jaccard', 'Δ Jaccard',
                             'Transformer', 'Complexity', 'Entropy']
    
    # Format numbers
    summary_table['Dice'] = summary_table['Dice'].apply(lambda x: f'{x:.4f}')
    summary_table['Δ Dice'] = summary_table['Δ Dice'].apply(lambda x: f'{x:+.4f}')
    summary_table['Jaccard'] = summary_table['Jaccard'].apply(lambda x: f'{x:.4f}')
    summary_table['Δ Jaccard'] = summary_table['Δ Jaccard'].apply(lambda x: f'{x:+.4f}')
    
    # Convert boolean to checkmark
    summary_table['Transformer'] = summary_table['Transformer'].apply(lambda x: '✓' if x else '✗')
    summary_table['Complexity'] = summary_table['Complexity'].apply(lambda x: '✓' if x else '✗')
    summary_table['Entropy'] = summary_table['Entropy'].apply(lambda x: '✓' if x else '✗')
    
    # Save to CSV
    csv_path = f'{output_dir}/ablation_summary.csv'
    summary_table.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Save to markdown
    md_path = f'{output_dir}/ablation_summary.md'
    with open(md_path, 'w') as f:
        f.write("# Ablation Study Results Summary\n\n")
        f.write(summary_table.to_markdown(index=False))
        f.write("\n\n## Key Findings:\n\n")
        
        # Best configuration
        best_config = summary_table.iloc[0]
        f.write(f"- **Best Configuration**: `{best_config['Configuration']}` with Dice={best_config['Dice']}\n")
        
        # Component importance
        f.write("\n### Component Importance:\n\n")
        
        # Transformer
        with_trans = df[df['has_transformer']]['dice'].mean()
        without_trans = df[~df['has_transformer']]['dice'].mean()
        f.write(f"- **Transformer**: {with_trans:.4f} (with) vs {without_trans:.4f} (without) = {with_trans - without_trans:+.4f}\n")
        
        # Complexity
        with_comp = df[df['has_complexity']]['dice'].mean()
        without_comp = df[~df['has_complexity']]['dice'].mean()
        f.write(f"- **Complexity**: {with_comp:.4f} (with) vs {without_comp:.4f} (without) = {with_comp - without_comp:+.4f}\n")
        
        # Entropy
        with_ent = df[df['has_entropy']]['dice'].mean()
        without_ent = df[~df['has_entropy']]['dice'].mean()
        f.write(f"- **Entropy**: {with_ent:.4f} (with) vs {without_ent:.4f} (without) = {with_ent - without_ent:+.4f}\n")
    
    print(f"✓ Saved: {md_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(summary_table.to_string(index=False))
    print("="*80)


if __name__ == '__main__':
    print("="*80)
    print("ABLATION STUDY ANALYSIS")
    print("="*80)
    print()
    
    # Thu thập kết quả
    print("Step 1: Collecting results from experiments...")
    df = collect_ablation_results()
    
    if df is None or len(df) == 0:
        print("ERROR: No results found! Please run ablation experiments first.")
        exit(1)
    
    print(f"\n✓ Found {len(df)} configurations\n")
    
    # Vẽ biểu đồ
    print("Step 2: Generating comparison plots...")
    plot_ablation_comparison(df)
    print()
    
    # Tạo bảng tổng kết
    print("Step 3: Generating summary tables...")
    generate_summary_table(df)
    print()
    
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nResults saved in: ./ablation_analysis/")
    print("  - ablation_metrics_comparison.png")
    print("  - ablation_contribution_analysis.png")
    print("  - ablation_summary.csv")
    print("  - ablation_summary.md")
