# Pareto V2 Multi-Objective NAS cho HCT-Net

## Tổng quan

Phiên bản cải tiến với **3 mục tiêu tối ưu hóa Pareto**:

1. **Accuracy (Dice Loss)** - Độ chính xác phân đoạn
2. **Complexity Loss** - FLOPs và Parameters của Transformer (**TRỌNG SỐ CAO**)
3. **Connection Differentiation** - Phân hóa rõ ràng ON/OFF cho transformer connections

## Tính năng mới

### 1. NAS cho Transformer Hyperparameters

Code tự động search các thông số tốt nhất cho từng transformer block:

- **d_model**: [64, 128, 256, 512] - Hidden dimension
- **n_head**: [2, 4, 8] - Số attention heads
- **expansion**: [2, 4, 8] - FFN expansion ratio

Tổng cộng có **30+ configurations** hợp lệ được search.

### 2. Complexity Loss với Trọng số CAO

```python
# Tính toán FLOPs và Parameters cho mỗi configuration
def calculate_transformer_complexity(config, seq_len=256, num_layers=6):
    - Multi-head attention FLOPs
    - FFN FLOPs
    - Parameters count
    - Trả về: (flops_millions, params_millions)
```

**Trọng số mặc định**:
- Dice loss: 1.0 (baseline)
- Complexity loss: **5.0** (HIGH - để force differentiation)
- Connection loss: 2.0 (moderate)

### 3. Pareto Multi-Objective Optimization

```python
def compute_pareto_loss(dice_loss, complexity_loss, connection_loss, args):
    # Normalize các loss về cùng scale [0, 1]
    # Weighted sum với trọng số khác nhau
    total_loss = (
        w_dice * dice_normalized +
        w_complexity * complexity_normalized +  # HIGH WEIGHT
        w_connection * connection_normalized
    )
```

### 4. Connection Differentiation Loss

Ngăn chặn tình trạng "all-on" hoặc "all-off":

```python
def compute_connection_differentiation_loss(model, args, epoch=None):
    # Penalty nếu tất cả connections giống nhau
    # Encourage variance cao trong probabilities
    # Warmup: giảm penalty trong early epochs
```

## Cách sử dụng

### Bước 1: Cập nhật Model

Thêm `alphas_transformer_configs` vào model class:

```python
# Trong _init_arch_parameters():
num_connections = (self.layers - 1) // 2
self.alphas_transformer_configs = nn.Parameter(
    1e-3 * torch.randn(num_connections, NUM_TRANSFORMER_CONFIGS)
)

# Thêm vào _arch_parameters list
self._arch_parameters = [
    self.alphas_down,
    self.alphas_up,
    self.alphas_normal,
    self.alphas_network,
    self.alphas_transformer_connections,
    self.alphas_transformer_configs,  # NEW
]
```

### Bước 2: Chạy Training

```bash
cd /mnt/data/KHTN2023/research25/hct-netm/hct_net

python train_CVCDataset_pareto_v2.py \
    --dataset cvc \
    --dataset_root ../datasets/cvc \
    --layers 9 \
    --epochs 50 \
    --arch_after 5 \
    --transformer_warmup_epochs 5 \
    --pareto_weight_dice 1.0 \
    --pareto_weight_complexity 5.0 \
    --pareto_weight_connection 2.0 \
    --train_batch 2 \
    --val_batch 2
```

### Bước 3: Điều chỉnh Hyperparameters

**Để tăng differentiation giữa các connections:**

```bash
# Tăng trọng số complexity
--pareto_weight_complexity 10.0  # Thay vì 5.0

# Tăng connection differentiation weight
--pareto_weight_connection 3.0  # Thay vì 2.0

# Tăng warmup epochs
--transformer_warmup_epochs 10  # Cho model thời gian explore
```

**Để giảm complexity (model nhỏ hơn):**

```bash
# Tăng complexity weight rất cao
--pareto_weight_complexity 20.0

# Giảm dice weight (chấp nhận accuracy thấp hơn một chút)
--pareto_weight_dice 0.5
```

## Kiến trúc Search Space

### Transformer Configurations (30+ choices)

Mỗi transformer connection có thể chọn 1 trong 30+ configs:

```
Config 0:  d_model=64,  n_head=2, expansion=2  → FLOPs=1.5M,  Params=0.1M
Config 1:  d_model=64,  n_head=2, expansion=4  → FLOPs=2.8M,  Params=0.2M
Config 2:  d_model=64,  n_head=2, expansion=8  → FLOPs=5.4M,  Params=0.4M
Config 3:  d_model=64,  n_head=4, expansion=2  → FLOPs=1.5M,  Params=0.1M
...
Config 28: d_model=512, n_head=8, expansion=4  → FLOPs=180M, Params=12M
Config 29: d_model=512, n_head=8, expansion=8  → FLOPs=360M, Params=24M
```

### Ví dụ Kết quả Search

```
Connection 0 (deepest):
  Config: d_model=256, n_head=8, expansion=4
  Complexity: FLOPs=45.2M, Params=3.1M
  State: ON (prob=0.95)

Connection 1 (middle):
  Config: d_model=128, n_head=4, expansion=2
  Complexity: FLOPs=11.3M, Params=0.8M
  State: ON (prob=0.78)

Connection 2 (shallowest):
  Config: d_model=64, n_head=2, expansion=2
  Complexity: FLOPs=1.5M, Params=0.1M
  State: OFF (prob=0.23)

Connection 3:
  Config: d_model=128, n_head=4, expansion=4
  Complexity: FLOPs=22.1M, Params=1.5M
  State: OFF (prob=0.15)

Total: 2/4 transformers ON
Total Complexity: 56.5M FLOPs, 3.9M Params
```

## Logging và Monitoring

### WandB Metrics

Tự động log các metrics sau:

```python
wandb.log({
    'train/dice': md,
    'train/complexity_loss': complexity_loss_avg,
    'train/connection_loss': connection_loss_avg,
    'architecture/trans_count_on': trans_on_count,
    'architecture/config_complexity': expected_complexity,
})
```

### TensorBoard

```bash
tensorboard --logdir=./search_exp/UnetLayer9/cvc/pareto_v2_*/tbx_log
```

### Console Output

Mỗi 5 epochs in ra:

```
Epoch:25 WeightLoss:0.234  ArchLoss:1.567
         Acc:0.985   Dice:0.912  Jc:0.842
         ComplexityLoss:45.23  ConnectionLoss:0.156
  Transformer Connections: 2/4 ON
    Probabilities: ['0.950', '0.780', '0.230', '0.150']
  Transformer Configurations:
    Connection 0: Config 18 (prob=0.923)
      d_model=256, n_head=8, expansion=4
      FLOPs=45.20M, Params=3.10M
```

## So sánh với Version Cũ

| Feature | Old Version | Pareto V2 |
|---------|-------------|-----------|
| Objectives | 1 (Dice only) | **3 (Dice + Complexity + Connection)** |
| Transformer Params | Fixed (d_model=256, n_head=4) | **Searchable (30+ configs)** |
| Complexity Awareness | ❌ None | **✅ FLOPs + Params calculated** |
| Differentiation | Weak (often all-on/all-off) | **Strong (high complexity weight)** |
| Connection Control | Random | **Pareto-guided** |

## Kết quả Mong đợi

### Với High Complexity Weight (5.0-10.0):

- ✅ Transformer connections phân hóa rõ: một số ON, một số OFF
- ✅ Configs khác nhau cho mỗi connection (không uniform)
- ✅ Connections sâu (gần output) có xu hướng ON, configs lớn hơn
- ✅ Connections nông (gần input) có xu hướng OFF hoặc configs nhỏ
- ✅ Trade-off tốt giữa accuracy và complexity

### Với Very High Complexity Weight (>10.0):

- ⚠️ Có thể hy sinh một chút accuracy
- ✅ Model rất nhỏ, chỉ 1-2 transformers ON
- ✅ Chọn configs nhỏ nhất (d_model=64, expansion=2)
- ✅ Phù hợp cho edge devices

## Troubleshooting

### Vấn đề: Tất cả connections đều ON

**Giải pháp:**
```bash
# Tăng complexity weight
--pareto_weight_complexity 10.0

# Tăng connection differentiation weight
--pareto_weight_connection 5.0
```

### Vấn đề: Tất cả connections đều OFF

**Giải pháp:**
```bash
# Giảm complexity weight
--pareto_weight_complexity 2.0

# Tăng dice weight
--pareto_weight_dice 2.0
```

### Vấn đề: Configs không converge

**Giải pháp:**
```bash
# Tăng warmup epochs
--transformer_warmup_epochs 10

# Giảm architecture learning rate
--arch_lr 5e-4

# Tăng số epochs
--epochs 100
```

## Advanced: Custom Configuration Space

Để thêm/sửa configuration choices:

```python
# Trong train_CVCDataset_pareto_v2.py
TRANSFORMER_D_MODEL_CHOICES = [32, 64, 128, 256, 512, 1024]  # Thêm 32 và 1024
TRANSFORMER_N_HEAD_CHOICES = [1, 2, 4, 8, 16]  # Thêm 1 và 16
TRANSFORMER_EXPANSION_CHOICES = [1, 2, 4, 8, 16]  # Thêm 1 và 16
```

Sau đó code tự động tạo lại search space với ~100+ configs.

## Citation

Nếu bạn sử dụng code này, vui lòng cite:

```
Pareto V2 Multi-Objective NAS for Transformer-based Medical Image Segmentation
- 3 objectives: Accuracy, Complexity, Connection Differentiation
- NAS for transformer hyperparameters (d_model, n_head, expansion)
- High complexity weight to force differentiated transformer selections
```

## TODO / Future Work

- [ ] Thêm objective thứ 4: Inference latency (thực tế đo trên GPU)
- [ ] Implement Pareto frontier tracking và visualization
- [ ] Auto-tune weights dựa trên validation performance
- [ ] Mixed precision training để tăng tốc
- [ ] Knowledge distillation từ large configs sang small configs

## Contact

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue hoặc liên hệ.
