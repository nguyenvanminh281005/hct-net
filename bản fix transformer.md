# Probability Differentiation Methods

## Vấn đề
Khi training NAS với nhiều connections, các probability `p_i` thường hội tụ về cùng một giá trị (ví dụ: tất cả đều ~0.5 hoặc tất cả đều ~0.8). Điều này làm mất khả năng phân biệt giữa các connections và không tạo ra kiến trúc tối ưu.

## 4 Cách Giải Quyết

### ✅ Cách 1: Tách Complexity Theo Từng Connection (BẮT BUỘC)

**Vấn đề với cách cũ:**
```python
f2 = total_flops(model)  # Tổng complexity toàn mô hình
```
→ Gradient giống nhau cho tất cả `p_i`, không khuyến khích phân hóa

**Giải pháp:**
```python
# Tính complexity riêng cho từng connection
f2 = sum_i (p_i * cost_i)  # cost_i khác nhau → gradient khác nhau
```

**Trong code:**
- `per_connection_complexity[]` được tính riêng cho từng connection
- `total_complexity = sum(per_connection_complexity)`
- Mỗi connection có gradient riêng dựa trên cost riêng của nó

**Tại sao hiệu quả:**
- Nếu `cost_i > cost_j`, thì gradient của `p_i` sẽ lớn hơn `p_j`
- Optimizer tự nhiên sẽ ưu tiên giảm `p_i` (connection đắt) hơn `p_j` (connection rẻ)
- Tạo ra sự phân hóa tự nhiên dựa trên cost

---

### ✅ Cách 2: Thay Entropy Bằng Variance/Repulsion Loss

**Vấn đề với Entropy:**
```python
H(p) = -Σ p_i log(p_i)  # Entropy cao khi distribution uniform
```
→ Không cấm các `p_i` bằng nhau, chỉ khuyến khích uncertainty

**Giải pháp 2a: Variance Loss**
```python
L_div = -Var(p) = -mean((p_i - mean(p))^2)
```
- Maximize variance → minimize negative variance
- Variance cao = các `p_i` khác nhau nhiều ✅
- Variance thấp = các `p_i` giống nhau ❌

**Giải pháp 2b: Repulsion Loss**
```python
L_div = sum_{i≠j} exp(-|p_i - p_j|)
```
- Penalize khi `p_i` và `p_j` quá gần nhau
- Force các connections có probability khác nhau
- Mạnh hơn variance loss nhưng tốn computation hơn

**Sử dụng:**
```bash
--diversity_loss_type variance   # Hoặc 'repulsion' hoặc 'indecision'
```

---

### ✅ Cách 3: Inject Positional Bias (Phá Symmetry)

**Vấn đề:**
Nếu tất cả connections khởi tạo giống nhau và có cùng cost → chúng sẽ học giống nhau

**Giải pháp 3a: Positional Initialization Bias**
```python
# Thay vì:
alpha_i = bias  # Cùng giá trị cho tất cả

# Làm:
alpha_i = bias + ε * i  # Mỗi position có initialization khác nhau
```

**Giải pháp 3b: Positional Cost Bias**
```python
cost_i = base_cost * (1 + γ * i)  # Cost tăng theo depth
```

**Sử dụng:**
```bash
--positional_init_scale 0.15      # ε: scale for alpha initialization
--positional_bias_factor 0.1      # γ: scale for cost bias
```

**Tại sao hiệu quả:**
- Phá vỡ symmetry ngay từ đầu
- Connections ở vị trí khác nhau có "identity" khác nhau
- Ngay cả khi structure giống nhau, position bias tạo sự khác biệt

---

### ✅ Cách 4: Gumbel-Softmax Discrete Sampling

**Vấn đề với Soft Bernoulli:**
```python
p = softmax(alpha)  # Smooth, continuous
```
→ Dễ hội tụ về giá trị uniform (tất cả ~0.5)

**Giải pháp: Gumbel-Softmax**
```python
def gumbel_softmax(logits, temperature):
    # Add Gumbel noise
    gumbel_noise = -log(-log(uniform_random))
    y = softmax((logits + gumbel_noise) / temperature)
    return y
```

**Properties:**
- Temperature cao (τ→∞): giống softmax (smooth)
- Temperature thấp (τ→0): giống argmax (discrete)
- Temperature annealing: bắt đầu smooth, dần dần discrete

**Sử dụng:**
```bash
--use_gumbel_softmax                # Enable Gumbel-Softmax
--gumbel_temperature 1.0            # Initial temperature
--gumbel_anneal                     # Enable annealing
--gumbel_temp_min 0.5               # Minimum temperature
```

**Tại sao hiệu quả:**
- Gumbel noise tạo randomness → tránh local minima
- Hard assignment (với hard=True) → force discrete decisions
- Annealing → smooth training ban đầu, discrete decisions sau

---

## Khuyến Nghị Sử Dụng

### Quick Start (Single Method)

**Nếu chỉ chọn 1 cách:**
```bash
# Cách 3: Positional Bias (đơn giản nhất, hiệu quả)
python train_CVCDataset_pareto_v2.py \
    --positional_init_scale 0.15 \
    --positional_bias_factor 0.1
```

**Nếu chỉ chọn 2 cách:**
```bash
# Cách 1 (auto) + Cách 2 (Variance)
python train_CVCDataset_pareto_v2.py \
    --diversity_loss_type variance \
    --positional_bias_factor 0.05
```

### Recommended (All Methods Combined)

```bash
python train_CVCDataset_pareto_v2.py \
    --layers 9 \
    --epochs 15 \
    --diversity_loss_type variance \
    --positional_init_scale 0.15 \
    --positional_bias_factor 0.1 \
    --use_gumbel_softmax \
    --gumbel_temperature 1.0 \
    --gumbel_anneal \
    --gumbel_temp_min 0.5 \
    --pareto_weight_dice 0.3 \
    --pareto_weight_complexity 0.3 \
    --pareto_weight_connection 0.4
```

### Run All Experiments

```bash
chmod +x run_probability_differentiation.sh
./run_probability_differentiation.sh
```

---

## Tham Số Chi Tiết

### Diversity Loss (`--diversity_loss_type`)
- `indecision`: Original (penalize prob~0.5)
- `variance`: Variance loss (recommended)
- `repulsion`: Repulsion loss (strongest, slower)

### Positional Bias
- `--positional_init_scale`: ε cho alpha initialization (khuyến nghị: 0.1-0.2)
- `--positional_bias_factor`: γ cho cost bias (khuyến nghị: 0.05-0.1)

### Gumbel-Softmax
- `--use_gumbel_softmax`: Enable/disable
- `--gumbel_temperature`: Initial τ (khuyến nghị: 1.0)
- `--gumbel_anneal`: Enable temperature annealing
- `--gumbel_temp_min`: Minimum τ (khuyến nghị: 0.3-0.5)

### Pareto Weights
- `--pareto_weight_dice`: Weight cho accuracy (0.3)
- `--pareto_weight_complexity`: Weight cho efficiency (0.3)
- `--pareto_weight_connection`: Weight cho diversity (0.4)

---

## Kết Quả Mong Đợi

### Trước khi áp dụng:
```
Connection probabilities:
  [0.51, 0.52, 0.51, 0.50, 0.51]  ❌ Too similar!
Std: 0.007
```

### Sau khi áp dụng:
```
Connection probabilities:
  [0.85, 0.23, 0.67, 0.12, 0.91]  ✅ Diverse!
Std: 0.335
```

---

## Lý Thuyết

### Tại sao probabilities hội tụ về cùng giá trị?

1. **Symmetric architecture**: Tất cả connections có cùng structure
2. **Shared gradient**: Loss không phân biệt giữa các connections
3. **Uniform initialization**: Bắt đầu từ cùng giá trị
4. **Mean-field approximation**: Optimizer tìm global minimum = uniform distribution

### Tại sao 4 cách này work?

**Cách 1** → Phá vỡ shared gradient (mỗi connection có gradient riêng)
**Cách 2** → Explicitly penalize uniformity trong loss
**Cách 3** → Phá vỡ symmetric initialization
**Cách 4** → Inject stochasticity để tránh local minima

---

## Code Implementation

### Cách 1: Per-Connection Complexity
```python
# File: train_CVCDataset_pareto_v2.py, line ~250
per_connection_complexity = []
for conn_idx in range(num_connections):
    # Each connection has its own complexity
    expected_complexity = sum(prob * cost for prob, cost in zip(config_probs, costs))
    
    # Apply positional bias (Cách 3)
    positional_multiplier = 1.0 + positional_bias_factor * conn_idx
    biased_complexity = expected_complexity * positional_multiplier
    
    # Cách 1: f2_i = p_i * cost_i
    weighted = probs_on[conn_idx] * biased_complexity
    per_connection_complexity.append(weighted)

total_complexity = sum(per_connection_complexity)
```

### Cách 2: Variance Loss
```python
# File: train_CVCDataset_pareto_v2.py, line ~325
if diversity_loss_type == 'variance':
    mean_prob = torch.mean(probs_on)
    variance = torch.mean((probs_on - mean_prob) ** 2)
    normalized_variance = variance / 0.25  # Max variance for binary
    f3 = 1.0 - normalized_variance  # Minimize = encourage high variance
```

### Cách 3: Positional Bias
```python
# File: train_CVCDataset_pareto_v2.py, line ~625
for i in range(len(alphas)):
    positional_offset = positional_init_scale * i
    alphas[i, 0] = base_off_logit - positional_offset
    alphas[i, 1] = base_on_logit + positional_offset
```

### Cách 4: Gumbel-Softmax
```python
# File: train_CVCDataset_pareto_v2.py, line ~35
def gumbel_softmax_sample(logits, temperature, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if hard:
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    
    return y
```

---

## Troubleshooting

### Nếu probabilities vẫn giống nhau:

1. **Tăng positional bias:**
   ```bash
   --positional_init_scale 0.2  # Tăng từ 0.15
   --positional_bias_factor 0.15  # Tăng từ 0.1
   ```

2. **Dùng repulsion loss thay vì variance:**
   ```bash
   --diversity_loss_type repulsion
   ```

3. **Tăng Gumbel temperature ban đầu:**
   ```bash
   --gumbel_temperature 2.0  # Tăng từ 1.0
   ```

4. **Tăng weight cho diversity loss:**
   ```bash
   --pareto_weight_connection 0.5  # Tăng từ 0.4
   ```

### Nếu training không stable:

1. **Giảm positional bias:**
   ```bash
   --positional_init_scale 0.05
   --positional_bias_factor 0.03
   ```

2. **Không dùng Gumbel annealing:**
   ```bash
   # Bỏ --gumbel_anneal
   ```

3. **Giảm diversity weight:**
   ```bash
   --pareto_weight_connection 0.2
   ```

---

## References

- **Cách 1**: Similar to NAS-Bench-201 per-op cost weighting
- **Cách 2**: Based on diversity regularization in multi-task learning
- **Cách 3**: Inspired by positional encoding in Transformers
- **Cách 4**: Gumbel-Softmax from Jang et al. (2017) "Categorical Reparameterization with Gumbel-Softmax"

---

## Monitoring

Trong training logs, xem:

```
Transformer connection initialization with positional bias:
  Connection 0 probs: OFF=0.426, ON=0.574
  Connection 1 probs: OFF=0.401, ON=0.599
  Connection 2 probs: OFF=0.376, ON=0.624
  Mean ON probability: 0.599
  Std ON probability: 0.025 (higher std = more diversity) ✅
```

Sau vài epochs:
```
Transformer Connections: 3/4 ON
  Probabilities: ['0.891', '0.234', '0.782', '0.156'] ✅ Diverse!
```

---

Created: 2025-12-26
Updated: 2025-12-26
Author: NAS Research Team
