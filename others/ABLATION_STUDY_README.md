# Hướng dẫn Ablation Study - HCT-Net

## Mục đích
Khảo sát sự cần thiết của 3 thành phần loss trong mô hình:
- **Transformer Loss**: Điều khiển transformer connections
- **Complexity Loss**: Tối ưu độ phức tạp operations
- **Entropy Loss**: Đảm bảo kiến trúc decisive

**Lưu ý**: Dice loss luôn có trong tất cả các trường hợp (baseline metric cho medical segmentation)

---

## 8 Trường hợp Ablation (2³ = 8)

| # | Mode | Dice | Transformer | Complexity | Entropy | Mô tả |
|---|------|------|-------------|------------|---------|-------|
| 1 | `all` | ✓ | ✓ | ✓ | ✓ | **Full model** - Tất cả thành phần |
| 2 | `no_transformer` | ✓ | ✗ | ✓ | ✓ | Loại bỏ transformer loss |
| 3 | `no_complexity` | ✓ | ✓ | ✗ | ✓ | Loại bỏ complexity loss |
| 4 | `no_entropy` | ✓ | ✓ | ✓ | ✗ | Loại bỏ entropy loss |
| 5 | `only_transformer` | ✓ | ✓ | ✗ | ✗ | Chỉ Dice + Transformer |
| 6 | `only_complexity` | ✓ | ✗ | ✓ | ✗ | Chỉ Dice + Complexity |
| 7 | `only_entropy` | ✓ | ✗ | ✗ | ✓ | Chỉ Dice + Entropy |
| 8 | `none` | ✓ | ✗ | ✗ | ✗ | **Baseline** - Chỉ Dice |

---

## Cách sử dụng

### 1. Chạy toàn bộ 8 trường hợp (Khuyến nghị)

```bash
cd /mnt/data/KHTN2023/research25/hct-netm
chmod +x run_ablation_study.sh
./run_ablation_study.sh
```

Script sẽ tự động chạy tất cả 8 cấu hình tuần tự.

---

### 2. Chạy từng trường hợp riêng lẻ

#### Trường hợp 1: Full model (ALL)
```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode all \
    --dataset cvc \
    --dataset_root /mnt/data/KHTN2023/research25/hct-netm/datasets/cvc \
    --epochs 15 \
    --train_batch 4 \
    --val_batch 4 \
    --layers 9
```

#### Trường hợp 2: No Transformer
```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode no_transformer \
    --dataset cvc \
    --epochs 15
```

#### Trường hợp 3: No Complexity
```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode no_complexity \
    --dataset cvc \
    --epochs 15
```

#### Trường hợp 4: No Entropy
```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode no_entropy \
    --dataset cvc \
    --epochs 15
```

#### Trường hợp 5: Only Transformer
```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode only_transformer \
    --dataset cvc \
    --epochs 15
```

#### Trường hợp 6: Only Complexity
```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode only_complexity \
    --dataset cvc \
    --epochs 15
```

#### Trường hợp 7: Only Entropy
```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode only_entropy \
    --dataset cvc \
    --epochs 15
```

#### Trường hợp 8: Baseline (NONE)
```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode none \
    --dataset cvc \
    --epochs 15
```

---

### 3. Override thủ công (Advanced)

Bạn có thể override từng thành phần:

```bash
python hct_net/train_CVCDataset.py \
    --ablation_mode custom \
    --use_transformer_loss True \
    --use_complexity_loss_component False \
    --use_entropy_loss True \
    --dataset cvc \
    --epochs 15
```

---

## Kết quả

### 1. Logs được lưu trong:
```
./search_exp/UnetLayer9/cvc/ablation_<mode>_<timestamp>/
```

Mỗi mode sẽ có folder riêng:
- `ablation_all_20231218-120000/`
- `ablation_no_transformer_20231218-130000/`
- `ablation_none_20231218-140000/`
- ...

### 2. Weights & Biases (wandb)

Project: `hct-net-ablation`

Tags tự động:
- Mode: `all`, `no_transformer`, `none`, etc.
- Component status: `dice_on`, `trans_on`, `comp_off`, `ent_on`, etc.

Truy cập: https://wandb.ai/your-account/hct-net-ablation

---

## So sánh kết quả

### Metrics quan trọng cần so sánh:

1. **Dice Score** (↑ cao hơn = tốt hơn)
2. **Jaccard Index** (↑ cao hơn = tốt hơn)
3. **Accuracy** (↑ cao hơn = tốt hơn)
4. **Model Complexity** (operations cost)
5. **Architecture Entropy** (decisiveness)
6. **Training Time**

### Phân tích đề xuất:

```python
import pandas as pd
import matplotlib.pyplot as plt

# So sánh Dice Score
results = {
    'all': 0.XX,
    'no_transformer': 0.XX,
    'no_complexity': 0.XX,
    'no_entropy': 0.XX,
    'only_transformer': 0.XX,
    'only_complexity': 0.XX,
    'only_entropy': 0.XX,
    'none': 0.XX,  # baseline
}

# Vẽ biểu đồ
df = pd.DataFrame.from_dict(results, orient='index', columns=['Dice Score'])
df.plot(kind='bar', title='Ablation Study: Impact on Dice Score')
plt.ylabel('Dice Score')
plt.xlabel('Configuration')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ablation_results.png')
```

### Câu hỏi nghiên cứu:

1. **Transformer Loss có cần thiết?**
   - So sánh: `all` vs `no_transformer`
   - So sánh: `only_transformer` vs `none`

2. **Complexity Loss có cần thiết?**
   - So sánh: `all` vs `no_complexity`
   - So sánh: `only_complexity` vs `none`

3. **Entropy Loss có cần thiết?**
   - So sánh: `all` vs `no_entropy`
   - So sánh: `only_entropy` vs `none`

4. **Kết hợp nào tốt nhất?**
   - So sánh tất cả 8 cấu hình
   - Xác định top 3 configurations

---

## Expected Runtime

- **Mỗi trường hợp**: ~X giờ (tùy vào GPU)
- **Toàn bộ 8 trường hợp**: ~8X giờ

**Khuyến nghị**: Chạy trên GPU mạnh hoặc chạy song song trên nhiều GPU.

---

## Troubleshooting

### Lỗi: "no module named wandb"
```bash
pip install wandb
wandb login
```

### Lỗi: "Dataset not found"
Kiểm tra đường dẫn dataset:
```bash
ls /mnt/data/KHTN2023/research25/hct-netm/datasets/cvc
```

### Lỗi: Out of memory
Giảm batch size:
```bash
python hct_net/train_CVCDataset.py --ablation_mode all --train_batch 2 --val_batch 2
```

---

## Tài liệu tham khảo

- Code chính: `hct_net/train_CVCDataset.py`
- Function ablation: `configure_ablation_study()` (line ~30)
- Hierarchical Pareto: `hierarchical_pareto_optimization()` (line ~187)
- Loss computation: `compute_pareto_guided_loss()` (line ~387)

---

**Tác giả**: Research Team  
**Ngày tạo**: December 18, 2025  
**Version**: 1.0
