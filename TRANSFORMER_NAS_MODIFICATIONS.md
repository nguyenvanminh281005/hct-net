# Transformer Layer NAS Implementation

## Tổng quan
Đây là bộ thay đổi để biến số lượng layer của Transformer thành searchable parameter trong Neural Architecture Search framework của HCT-Net.

## Các thay đổi chính

### 1. Genotypes.py
- **Thêm `TransformerLayerConfigs`**: Định nghĩa search space cho số layer của transformer
```python
TransformerLayerConfigs = [1, 2, 3, 4, 6, 8, 12]  # 7 configurations to choose from
```

### 2. train_CVCDataset.py 
- **Import TransformerLayerConfigs**
- **Khởi tạo switches_transformer**: Tạo switches cho transformer search space
- **Update model initialization**: Truyền switches_transformer vào model
- **Thêm transformer alphas**: Initialize và zero gradients cho transformer alphas
- **Early architecture fixing**: Implement confidence-based fixing cho transformer layers
- **Logging**: Thêm logging cho transformer architecture weights

### 3. hybridCnnTransformer.py
- **Constructor updates**: 
  - Thêm `switches_transformer` parameter
  - Store switches_transformer as class attribute
  - Initialize `fix_arch_transformer_index` dictionary

- **Architecture Parameters**:
  - Thêm `alphas_transformer` vào `_init_arch_parameters()`
  - Update `_arch_parameters` list
  - Modify `alphas_dict()` và `load_alphas()` methods

- **Dynamic Transformer Builder**:
  ```python
  def _get_dynamic_transformer(self, d_model, dim_feedforward, layer_idx):
      # Dynamically create transformer based on alpha weights
      # Supports both training (sampling) and inference (argmax)
  ```

- **Forward Pass Updates**:
  - Replace static transformer calls với dynamic transformer calls
  - Add early fixing support trong forward method

### 4. nas_model/__init__.py
- **Update get_models function**: Add switches_transformer parameter to all model instantiations

## Cách hoạt động

### Search Phase
1. **Initialization**: Mỗi layer có alphas cho 7 transformer configurations
2. **Sampling**: Trong training, sample từ softmax của alphas  
3. **Architecture Update**: Update alphas dựa trên validation loss
4. **Early Fixing**: Fix transformer config khi confidence >= 0.3

### Architecture Selection
- **Training**: Multinomial sampling từ transformer alphas
- **Inference**: Argmax selection từ learned alphas
- **Child Generation**: Argmax khi `gen_max_child_flag=True`

## Search Space Expansion

### Trước
- Fixed 6 layers cho tất cả transformers
- Total search space: CNN operations only

### Sau  
- Searchable transformer layers: 1, 2, 3, 4, 6, 8, 12
- Total search space: CNN operations × Transformer configurations^(num_layers)
- Exponential expansion của search space

## Scheme chi tiết

### Data Flow
```
Input Image
    ↓
Stem Layers (CNN)
    ↓
Layer 0: CNN Cell + Dynamic Transformer (64D, config from alphas[0])
    ↓
Layer 1: CNN Cell + Dynamic Transformer (128D, config from alphas[1]) 
    ↓
Layer 2: CNN Cell + Dynamic Transformer (256D, config from alphas[2])
    ↓
... (continue with remaining layers)
    ↓
Output Segmentation
```

### Architecture Parameters
```python
alphas_transformer: [layers × num_configs]
# layers=7, num_configs=7
# Shape: [7, 7] representing choices for each layer depth
```

### Training Loop
1. **Weight Update Phase**: Update CNN weights và transformer weights
2. **Architecture Update Phase**: Update transformer layer selection alphas
3. **Progressive Fixing**: Fix high-confidence transformer configurations
4. **Validation**: Evaluate complete architecture

## Lợi ích

1. **Adaptive Complexity**: Model có thể học optimal depth cho mỗi layer
2. **Resource Efficiency**: Shallow transformers ở early layers, deep ở later layers
3. **Task-Specific**: Different medical tasks có thể cần different transformer depths
4. **Differentiable**: End-to-end learning của architecture và weights

## Cách sử dụng

### Training
```bash
python train_CVCDataset.py --model UnetLayer7 --layers 7 --epochs 50
```

### Key Parameters  
- `--arch_after 3`: Start transformer architecture search after 3 epochs
- `--early_fix_arch`: Enable progressive architecture fixing
- `--gen_max_child`: Generate child networks using argmax

### Monitoring
- Check tensorboard logs cho transformer alpha evolution
- Monitor validation metrics để assess architecture quality
- Watch for convergence của transformer alphas

## Troubleshooting

### Common Issues
1. **Memory**: Dynamic transformers có thể consume more memory
2. **Convergence**: Search space rất lớn, có thể cần more epochs
3. **Initialization**: Ensure proper random seed để reproducible results

### Solutions
- Reduce batch size nếu memory issues
- Increase search epochs hoặc adjust learning rates
- Use early fixing để reduce search space over time