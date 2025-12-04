"""
Script vẽ biểu đồ so sánh Acc, Dice, Jc của 2 thí nghiệm training.

Bảng số liệu đã trích xuất từ log files:

UnetLayer7 (20251019-221058) - 10 epochs:
Epoch | Acc   | Dice  | Jc
------|-------|-------|-------
0     | 0.747 | 0.525 | 0.434
1     | 0.830 | 0.729 | 0.626
2     | 0.850 | 0.746 | 0.646
3     | 0.859 | 0.757 | 0.660
4     | 0.863 | 0.768 | 0.672
5     | 0.868 | 0.782 | 0.691
6     | 0.877 | 0.776 | 0.687
7     | 0.884 | 0.794 | 0.706
8     | 0.885 | 0.788 | 0.703
9     | 0.860 | 0.735 | 0.651

Nas_Search_Unet (20251008-085714) - 5 epochs:
Epoch | Acc   | Dice  | Jc
------|-------|-------|-------
0     | 0.803 | 0.682 | 0.586
1     | 0.856 | 0.766 | 0.669
2     | 0.870 | 0.782 | 0.689
3     | 0.883 | 0.799 | 0.714
4     | 0.884 | 0.808 | 0.724
"""

import matplotlib.pyplot as plt
import argparse

# Dữ liệu trích xuất từ log files
DATA = {
    'UnetLayer7': {
        'epochs': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'acc':    [0.747, 0.830, 0.850, 0.859, 0.863, 0.868, 0.877, 0.884, 0.885, 0.860],
        'dice':   [0.525, 0.729, 0.746, 0.757, 0.768, 0.782, 0.776, 0.794, 0.788, 0.735],
        'jc':     [0.434, 0.626, 0.646, 0.660, 0.672, 0.691, 0.687, 0.706, 0.703, 0.651],
    },
    'Nas_Search_Unet': {
        'epochs': [0, 1, 2, 3, 4],
        'acc':    [0.803, 0.856, 0.870, 0.883, 0.884],
        'dice':   [0.682, 0.766, 0.782, 0.799, 0.808],
        'jc':     [0.586, 0.669, 0.689, 0.714, 0.724],
    }
}


def plot_comparison(output_path='compare_logs.png'):
    """Vẽ biểu đồ so sánh Acc, Dice, Jc từ bảng dữ liệu."""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot Acc
    axes[0].plot(DATA['UnetLayer7']['epochs'], DATA['UnetLayer7']['acc'], 
                 marker='o', linewidth=2, label='UnetLayer7', color='#1f77b4')
    axes[0].plot(DATA['Nas_Search_Unet']['epochs'], DATA['Nas_Search_Unet']['acc'], 
                 marker='s', linewidth=2, label='Nas_Search_Unet', color='#ff7f0e')
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].set_ylim([0.7, 0.9])
    
    # Plot Dice
    axes[1].plot(DATA['UnetLayer7']['epochs'], DATA['UnetLayer7']['dice'], 
                 marker='o', linewidth=2, label='UnetLayer7', color='#1f77b4')
    axes[1].plot(DATA['Nas_Search_Unet']['epochs'], DATA['Nas_Search_Unet']['dice'], 
                 marker='s', linewidth=2, label='Nas_Search_Unet', color='#ff7f0e')
    axes[1].set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].set_ylim([0.5, 0.85])
    
    # Plot Jc (Jaccard)
    axes[2].plot(DATA['UnetLayer7']['epochs'], DATA['UnetLayer7']['jc'], 
                 marker='o', linewidth=2, label='UnetLayer7', color='#1f77b4')
    axes[2].plot(DATA['Nas_Search_Unet']['epochs'], DATA['Nas_Search_Unet']['jc'], 
                 marker='s', linewidth=2, label='Nas_Search_Unet', color='#ff7f0e')
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Jaccard Index', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='lower right', fontsize=10)
    axes[2].set_ylim([0.4, 0.75])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ vào: {output_path}")
    print(f"\nThống kê:")
    print(f"  UnetLayer7 - Best: Acc={max(DATA['UnetLayer7']['acc']):.3f}, "
          f"Dice={max(DATA['UnetLayer7']['dice']):.3f}, Jc={max(DATA['UnetLayer7']['jc']):.3f}")
    print(f"  Nas_Search_Unet - Best: Acc={max(DATA['Nas_Search_Unet']['acc']):.3f}, "
          f"Dice={max(DATA['Nas_Search_Unet']['dice']):.3f}, Jc={max(DATA['Nas_Search_Unet']['jc']):.3f}")


def main():
    parser = argparse.ArgumentParser(description='Vẽ biểu đồ so sánh training metrics')
    parser.add_argument('-o', '--output', default='compare_logs.png', 
                        help='Đường dẫn file PNG output (mặc định: compare_logs.png)')
    args = parser.parse_args()
    plot_comparison(args.output)


if __name__ == '__main__':
    main()
