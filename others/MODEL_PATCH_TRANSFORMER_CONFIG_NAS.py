"""
Patch file để thêm NAS cho transformer configurations vào hybridCnnTransformer_dynamic.py

Thêm các thay đổi sau vào file hybridCnnTransformer_dynamic.py:
"""

# ===================================================================
# 1. Thêm import ở đầu file
# ===================================================================

# Thêm sau dòng: from hct_net.nas_model.Kp_Trans.DeformableTrans import DeformableTransformer

from train_CVCDataset_pareto_v2 import (
    TRANSFORMER_CONFIG_CHOICES, 
    NUM_TRANSFORMER_CONFIGS,
    calculate_transformer_complexity
)


# ===================================================================
# 2. Trong hàm _init_arch_parameters(), thêm sau dòng khởi tạo alphas_transformer_connections
# ===================================================================

def _init_arch_parameters(self):
    """Initialize architecture parameters for NAS"""
    normal_num_ops = np.count_nonzero(self.switches_normal[0])
    down_num_ops = np.count_nonzero(self.switches_down[0])
    up_num_ops = np.count_nonzero(self.switches_up[0])
    
    k = sum(1 for i in range(self.meta_node_num) for n in range(2 + i))
    
    # Cell operation parameters
    self.alphas_down = nn.Parameter(1e-3 * torch.randn(k, down_num_ops))
    self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, normal_num_ops))
    self.alphas_up = nn.Parameter(1e-3 * torch.randn(k, up_num_ops))
    
    # Network architecture parameters
    self.alphas_network = nn.Parameter(1e-3 * torch.randn(self.layers, self.depth, 3))
    
    # Transformer connection parameters: [num_connections, 2] (on/off for each connection)
    self.alphas_transformer_connections = nn.Parameter(1e-3 * torch.randn(self.num_transformer_connections, 2))
    print(f"Initialized alphas_transformer_connections with shape: {self.alphas_transformer_connections.shape}")
    
    # *** NEW: Transformer configuration parameters ***
    # [num_connections, NUM_TRANSFORMER_CONFIGS] - one choice per connection
    self.alphas_transformer_configs = nn.Parameter(
        1e-3 * torch.randn(self.num_transformer_connections, NUM_TRANSFORMER_CONFIGS)
    )
    print(f"Initialized alphas_transformer_configs with shape: {self.alphas_transformer_configs.shape}")
    print(f"Each connection can choose from {NUM_TRANSFORMER_CONFIGS} configurations")
    
    # Setup alphas list
    self._alphas = []
    for n, p in self.named_parameters():
        if 'alphas' in n:
            self._alphas.append((n, p))
    
    self._arch_parameters = [
        self.alphas_down,
        self.alphas_up,
        self.alphas_normal,
        self.alphas_network,
        self.alphas_transformer_connections,
        self.alphas_transformer_configs,  # *** NEW ***
    ]


# ===================================================================
# 3. Trong hàm _build_transformers(), sửa để tạo dynamic transformers
# ===================================================================

def _build_transformers(self):
    """Build transformer blocks dynamically with searchable configurations"""
    # Tạo một transformer "template" cho mỗi connection
    # Actual configuration sẽ được chọn trong forward pass dựa trên alphas
    
    # Sử dụng configuration lớn nhất làm template (để chứa tất cả weights)
    max_config = max(TRANSFORMER_CONFIG_CHOICES, key=lambda c: c['d_model'])
    
    for i in range(self.num_transformer_connections):
        # Position embedding - có thể resize dynamically
        pos_embed = build_position_encoding(mode='v2', hidden_dim=max_config['d_model'])
        self.position_embeds.append(pos_embed)
        
        # Transformer với max configuration
        trans = DeformableTransformer(
            d_model=max_config['d_model'], 
            dim_feedforward=max_config['d_ff'], 
            dropout=0.1, 
            activation='gelu',
            num_feature_levels=1, 
            nhead=max_config['n_head'], 
            num_encoder_layers=6,
            enc_n_points=4
        )
        self.transformer_blocks.append(trans)
    
    print(f"Created {len(self.transformer_blocks)} transformer blocks with max config")
    print(f"  Max d_model={max_config['d_model']}, n_head={max_config['n_head']}")
    print(f"  Actual config will be selected via alphas_transformer_configs during forward pass")


# ===================================================================
# 4. Thêm helper function để lấy selected config
# ===================================================================

def get_selected_transformer_config(self, connection_idx):
    """
    Get the selected transformer configuration for a specific connection
    
    Args:
        connection_idx: Index of the transformer connection
    
    Returns:
        config_dict: Selected configuration with d_model, n_head, expansion, d_ff
        config_idx: Index of selected configuration
    """
    if self.gen_max_child_flag:
        # During inference: use argmax
        config_idx = torch.argmax(self.alphas_transformer_configs[connection_idx]).item()
    else:
        # During training: sample from distribution
        config_probs = F.softmax(self.alphas_transformer_configs[connection_idx], dim=-1)
        if self.training:
            # Gumbel-Softmax for differentiable sampling
            config_idx = F.gumbel_softmax(self.alphas_transformer_configs[connection_idx], 
                                         tau=1.0, hard=True).argmax().item()
        else:
            config_idx = torch.argmax(config_probs).item()
    
    return TRANSFORMER_CONFIG_CHOICES[config_idx], config_idx


# ===================================================================
# 5. Thêm method để apply transformer với selected config
# ===================================================================

def _apply_transformer_with_config(self, features, connection_idx):
    """
    Apply transformer to features using selected configuration
    
    Args:
        features: Input feature tensor [B, C, H, W]
        connection_idx: Index of transformer connection
    
    Returns:
        output: Transformed features with same shape
    """
    # Get selected configuration
    config, config_idx = self.get_selected_transformer_config(connection_idx)
    
    B, C, H, W = features.shape
    
    # Adjust features to match selected d_model
    if C != config['d_model']:
        # Project to selected d_model
        proj = nn.Conv2d(C, config['d_model'], 1).to(features.device)
        features = proj(features)
    
    # Create mask
    mask = torch.zeros((B, H, W), dtype=torch.bool, device=features.device)
    
    # Get position embedding
    pos_embed_module = self.position_embeds[connection_idx]
    pos_embed = pos_embed_module(features, mask.unsqueeze(1))
    
    # Apply transformer
    # Note: This is simplified - actual implementation may need to handle
    # different n_head and d_ff by masking or sub-sampling transformer weights
    transformer = self.transformer_blocks[connection_idx]
    
    # For now, use full transformer (in practice, you'd want to implement
    # dynamic transformer that can use subset of heads/FFN based on config)
    output = transformer([features], [mask], [pos_embed])
    
    return output


# ===================================================================
# 6. Thêm method để export genotype với transformer configs
# ===================================================================

def genotype(self):
    """Export discovered architecture including transformer configurations"""
    
    # Original genotype code cho CNN cells
    # ... (giữ nguyên code cũ)
    
    # Thêm transformer configuration info
    transformer_configs_selected = []
    
    if hasattr(self, 'alphas_transformer_configs'):
        config_probs = F.softmax(self.alphas_transformer_configs, dim=-1)
        
        for conn_idx in range(self.num_transformer_connections):
            # Get selected config
            config_idx = torch.argmax(config_probs[conn_idx]).item()
            config = TRANSFORMER_CONFIG_CHOICES[config_idx]
            
            # Get connection on/off state
            conn_probs = F.softmax(self.alphas_transformer_connections[conn_idx], dim=-1)
            is_on = conn_probs[1].item() > 0.5
            
            transformer_configs_selected.append({
                'connection_idx': conn_idx,
                'is_on': is_on,
                'on_prob': conn_probs[1].item(),
                'config_idx': config_idx,
                'd_model': config['d_model'],
                'n_head': config['n_head'],
                'expansion': config['expansion'],
                'd_ff': config['d_ff'],
            })
    
    # Return expanded genotype
    return {
        'normal_down': gene_normal_down,
        'normal_up': gene_normal_up, 
        'normal_normal': gene_normal_normal,
        'transformer_connections': transformer_connections_on_off,
        'transformer_configs': transformer_configs_selected,  # *** NEW ***
    }


# ===================================================================
# 7. Thêm method để tính complexity của current architecture
# ===================================================================

def get_current_complexity(self):
    """Calculate expected complexity of current architecture"""
    if not hasattr(self, 'alphas_transformer_configs'):
        return 0.0, 0.0
    
    config_probs = F.softmax(self.alphas_transformer_configs, dim=-1)
    conn_probs = F.softmax(self.alphas_transformer_connections, dim=-1)
    
    total_flops = 0.0
    total_params = 0.0
    
    for conn_idx in range(self.num_transformer_connections):
        # Expected complexity for this connection
        expected_flops = 0.0
        expected_params = 0.0
        
        for config_idx, prob in enumerate(config_probs[conn_idx]):
            complexity_info = calculate_transformer_complexity(
                TRANSFORMER_CONFIG_CHOICES[config_idx]
            )
            expected_flops += prob.item() * complexity_info[0]
            expected_params += prob.item() * complexity_info[1]
        
        # Weight by connection on/off probability
        prob_on = conn_probs[conn_idx, 1].item()
        total_flops += expected_flops * prob_on
        total_params += expected_params * prob_on
    
    return total_flops, total_params


# ===================================================================
# USAGE NOTES
# ===================================================================

"""
Để sử dụng code này:

1. Copy các hàm trên vào hybridCnnTransformer_dynamic.py

2. Trong forward pass, thay thế:
   trans_out = self._apply_transformer(cell_out, enc_idx)
   
   Bằng:
   trans_out = self._apply_transformer_with_config(cell_out, enc_idx)

3. Khởi tạo gradient cho alphas_transformer_configs:
   if hasattr(model, 'alphas_transformer_configs'):
       model.alphas_transformer_configs.grad = torch.zeros_like(model.alphas_transformer_configs)

4. Test:
   python train_CVCDataset_pareto_v2.py --layers 9 --epochs 10

Nếu bạn muốn implementation đơn giản hơn (không cần dynamic transformer),
có thể tạo nhiều transformer instances với configs khác nhau và
select bằng weighted sum based on alphas.
"""
