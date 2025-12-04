import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch.autograd import Variable
from torch.nn import init

sys.path.append('D:/KHTN2023/research25/hct-netm')
from hct_net.cell import Cell
from hct_net.genotypes import *
from hct_net.operations import *
from hct_net.nas_model.Kp_Trans.position_encoding import build_position_encoding
from hct_net.nas_model.Kp_Trans.DeformableTrans import DeformableTransformer


class hybridCnnTransDynamic(nn.Module):
    """
    Dynamic Hybrid CNN-Transformer model with searchable transformer connections.
    Supports variable number of layers (3, 5, 7, 9, etc.)
    """
    def __init__(self, input_c=3, c=16, num_classes=1, meta_node_num=4, layers=7, dp=0,
                 use_sharing=True, double_down_channel=True, use_softmax_head=False,
                 switches_normal=[], switches_down=[], switches_up=[], early_fix_arch=False, 
                 gen_max_child_flag=False, random_sample=False):
        super(hybridCnnTransDynamic, self).__init__()
        
        # Basic parameters
        self.CellLinkDownPos = CellLinkDownPos
        self.CellPos = CellPos
        self.CellLinkUpPos = CellLinkUpPos
        self.switches_normal = switches_normal
        self.switches_down = switches_down
        self.switches_up = switches_up
        self.dropout_prob = dp
        self.input_c = input_c
        self.num_class = num_classes
        self.meta_node_num = meta_node_num
        self.layers = layers  # Can be 3, 5, 7, 9, etc.
        self.use_sharing = use_sharing
        self.double_down_channel = double_down_channel
        self.use_softmax_head = use_softmax_head
        self.depth = (self.layers + 1) // 2
        self.c_prev_prev = 32
        self.c_prev = 64
        self.early_fix_arch = early_fix_arch
        self.gen_max_child_flag = gen_max_child_flag
        self.random_sample = random_sample
        
        # Calculate number of transformer connections: (layers - 1) / 2
        self.num_transformer_connections = (self.layers - 1) // 2
        print(f"Initializing dynamic model with {self.layers} layers and {self.num_transformer_connections} transformer connections")
        
        # Stem layers
        self.stem0 = ConvOps(input_c, self.c_prev_prev, kernel_size=3, stride=1, ops_order='weight_norm_act')
        self.stem1 = ConvOps(self.c_prev_prev, self.c_prev, kernel_size=3, stride=2, ops_order='weight_norm_act')
        
        # Calculate channel sizes
        init_channel = c
        if self.double_down_channel:
            self.layers_channel = [self.meta_node_num * init_channel * pow(2, i) for i in range(self.depth)]
            self.cell_channels = [init_channel * pow(2, i) for i in range(self.depth)]
        else:
            self.layers_channel = [self.meta_node_num * init_channel for i in range(0, self.depth)]
            self.cell_channels = [init_channel for i in range(0, self.depth)]
        
        # Build cells dynamically
        self.cells = nn.ModuleDict()
        self._build_cells()
        
        # Build transformer blocks dynamically
        self.transformer_blocks = nn.ModuleList()
        self.position_embeds = nn.ModuleList()
        self._build_transformers()
        
        # Output layers - create for each scale in decoder
        self.output_layers = nn.ModuleDict()
        for layer_idx in range(2, self.layers, 2):  # Only decoder layers (even indices)
            self.output_layers[f'layer_{layer_idx}'] = ConvOps(
                self.layers_channel[0], num_classes, kernel_size=1, 
                dropout_rate=0.1, ops_order='weight'
            )
        
        if self.use_softmax_head:
            self.softmax = nn.LogSoftmax(dim=1)
        
        # Initialize architecture parameters
        self._init_arch_parameters()
        
        if self.early_fix_arch:
            self.fix_arch_down_index = {}
            self.fix_arch_normal_index = {}
            self.fix_arch_up_index = {}
    
    def _build_cells(self):
        """Dynamically build cells based on number of layers"""
        for i in range(1, self.layers):
            level = self._get_level(i)
            
            if i == 1:
                # First downsampling layer
                self.cells[f'cell_{i}_{level}'] = Cell(
                    self.meta_node_num, -1, self.c_prev, self.cell_channels[level],
                    switch_normal=self.switches_normal, switch_down=self.switches_down,
                    switch_up=self.switches_up, cell_type="normal_down", dp=self.dropout_prob
                )
            else:
                # Determine cell type and connections based on position
                self._build_cell_at_layer(i, level)
    
    def _build_cell_at_layer(self, layer_idx, level):
        """Build cells at a specific layer with proper connections"""
        # Calculate which layers are available as inputs
        prev_layers = []
        for prev_idx in range(1, layer_idx):
            prev_level = self._get_level(prev_idx)
            prev_layers.append((prev_idx, prev_level))
        
        # Build cells with different spatial resolutions at this layer
        for target_level in range(self.depth):
            # Skip if this combination doesn't make sense for the architecture
            if layer_idx == 2 and target_level > 1:
                continue
            if layer_idx >= self.depth and target_level > self.depth - (layer_idx - self.depth + 1):
                continue
            
            # Determine connections from previous layer
            possible_inputs = self._get_possible_inputs(layer_idx, target_level, prev_layers)
            
            for input_idx, (prev_idx, prev_level) in enumerate(possible_inputs[:3]):  # Max 3 connections
                cell_type = self._get_cell_type(prev_level, target_level)
                cell_name = f'cell_{layer_idx}_{target_level}_{input_idx}'
                
                # Get previous cell channels
                if prev_idx == 0:
                    c_prev = self.c_prev
                else:
                    c_prev = self.cell_channels[prev_level]
                
                c_curr = self.cell_channels[target_level]
                
                self.cells[cell_name] = Cell(
                    self.meta_node_num, -1, c_prev, c_curr,
                    switch_normal=self.switches_normal, 
                    switch_down=self.switches_down,
                    switch_up=self.switches_up, 
                    cell_type=cell_type, 
                    dp=self.dropout_prob
                )
    
    def _get_level(self, layer_idx):
        """Get the depth level for a given layer index"""
        if layer_idx <= self.depth:
            return layer_idx - 1
        else:
            return self.depth - (layer_idx - self.depth + 1)
    
    def _get_cell_type(self, from_level, to_level):
        """Determine cell type based on spatial resolution change"""
        if from_level == to_level:
            return "normal_normal"
        elif from_level < to_level:
            return "normal_down"
        else:
            return "normal_up"
    
    def _get_possible_inputs(self, layer_idx, target_level, prev_layers):
        """Get possible input layers for current layer"""
        possible = []
        for prev_idx, prev_level in prev_layers:
            # Can connect if within 1 level difference
            if abs(prev_level - target_level) <= 1:
                possible.append((prev_idx, prev_level))
        return possible
    
    def _build_transformers(self):
        """Build transformer blocks dynamically based on number of connections"""
        transformer_configs = [
            {'hidden_dim': 64, 'd_model': 64, 'dim_feedforward': 256},
            {'hidden_dim': 128, 'd_model': 128, 'dim_feedforward': 512},
            {'hidden_dim': 256, 'd_model': 256, 'dim_feedforward': 1024},
            {'hidden_dim': 512, 'd_model': 512, 'dim_feedforward': 2048},
        ]
        
        for i in range(self.num_transformer_connections):
            config = transformer_configs[min(i, len(transformer_configs) - 1)]
            
            # Position embedding
            pos_embed = build_position_encoding(mode='v2', hidden_dim=config['hidden_dim'])
            self.position_embeds.append(pos_embed)
            
            # Transformer
            trans = DeformableTransformer(
                d_model=config['d_model'], 
                dim_feedforward=config['dim_feedforward'], 
                dropout=0.1, 
                activation='gelu',
                num_feature_levels=1, 
                nhead=4, 
                num_encoder_layers=6,
                enc_n_points=4
            )
            self.transformer_blocks.append(trans)
        
        print(f"Created {len(self.transformer_blocks)} transformer blocks")
    
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
        ]
    
    def forward(self, input, target, criterion):
        """Dynamic forward pass that adapts to number of layers"""
        _, _, h, w = input.size()
        
        # Get weights for cell operations
        if self.gen_max_child_flag:
            self.weights_normal = torch.zeros_like(self.alphas_normal).scatter_(
                1, torch.argmax(self.alphas_normal, dim=-1).view(-1, 1), 1)
            self.weights_down = torch.zeros_like(self.alphas_down).scatter_(
                1, torch.argmax(self.alphas_down, dim=-1).view(-1, 1), 1)
            self.weights_up = torch.zeros_like(self.alphas_up).scatter_(
                1, torch.argmax(self.alphas_up, dim=-1).view(-1, 1), 1)
            self.network_weight = F.softmax(self.alphas_network, dim=-1)
            self.transformer_weights = torch.zeros_like(self.alphas_transformer_connections).scatter_(
                1, torch.argmax(self.alphas_transformer_connections, dim=-1).view(-1, 1), 1)
        else:
            self.weights_normal = self._get_weights(self.alphas_normal)
            self.weights_down = self._get_weights(self.alphas_down)
            self.weights_up = self._get_weights(self.alphas_up)
            self.network_weight = F.softmax(self.alphas_network, dim=-1)
            self.transformer_weights = self._get_weights(self.alphas_transformer_connections)
        
        # Apply early fix arch if needed
        if self.early_fix_arch:
            self._apply_fixed_arch()
        
        # Compute loss_alpha for gradient-based search
        loss_alpha = None
        if not self.random_sample and self.training and not self.gen_max_child_flag:
            loss_alpha = self._compute_alpha_loss()
        
        # Forward pass through stem
        stem0_f = self.stem0(input)
        stem1_f = self.stem1(stem0_f)
        
        # Store intermediate features
        features = {0: stem1_f}
        transformer_features = {}
        
        # Process encoder path with transformers
        encoder_layers = list(range(1, self.depth))
        for enc_idx, layer_idx in enumerate(encoder_layers):
            # Get cell output
            cell_key = f'cell_{layer_idx}_{layer_idx - 1}'
            if cell_key in self.cells:
                prev_feat = features[layer_idx - 1] if layer_idx > 1 else stem1_f
                cell_out = self.cells[cell_key](None, prev_feat, self.weights_normal, 
                                               self.weights_down, self.weights_up)
                features[layer_idx] = cell_out
                
                # Apply transformer if available
                if enc_idx < len(self.transformer_blocks):
                    trans_out = self._apply_transformer(cell_out, enc_idx)
                    # Apply gating based on transformer connection weights
                    gate = self.transformer_weights[enc_idx, 1]  # Index 1 is "on"
                    transformer_features[enc_idx] = trans_out * gate
        
        # Bottleneck
        bottleneck_idx = self.depth - 1
        
        # Decoder path with skip connections
        decoder_outputs = []
        for layer_idx in range(self.depth, self.layers):
            level = self._get_level(layer_idx)
            # Build decoder features with skip connections from encoder
            # This is a simplified version - you'll need to implement full logic
            pass
        
        # Collect outputs from multiple scales
        outputs = []
        for layer_idx in range(2, self.layers, 2):
            if f'layer_{layer_idx}' in self.output_layers:
                feat_key = layer_idx
                if feat_key in features:
                    out = self.output_layers[f'layer_{layer_idx}'](features[feat_key])
                    out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
                    outputs.append(out)
        
        # Handle gradient computation for architecture search
        # DISABLED: We handle architecture gradients externally in the training loop
        # to support Pareto multi-objective optimization
        # if not self.random_sample and self.training and not self.gen_max_child_flag and loss_alpha is not None:
        #     self._compute_architecture_gradients(outputs, target, criterion, loss_alpha)
        
        return outputs if len(outputs) > 0 else [F.interpolate(features[0], size=(h, w), mode='bilinear', align_corners=False)]
    
    def _apply_transformer(self, feature, trans_idx):
        """Apply transformer to feature map"""
        x_fea = [feature]
        x_posemb = [self.position_embeds[trans_idx](feature)]
        masks = [torch.zeros((feature.shape[0], feature.shape[2], feature.shape[3]), 
                            dtype=torch.bool, device=feature.device)]
        
        x_trans = self.transformer_blocks[trans_idx](x_fea, masks, x_posemb)
        
        # Reshape back to feature map
        # Calculate output size based on feature dimensions
        output_size = feature.shape[0] * feature.shape[2] * feature.shape[3]
        trans_out = x_trans[:, :output_size, :].transpose(-1, -2).view(feature.shape)
        
        return trans_out
    
    def _apply_fixed_arch(self):
        """Apply fixed architecture for early stopping"""
        if len(self.fix_arch_down_index.keys()) > 0:
            for key, value_lst in self.fix_arch_down_index.items():
                self.weights_down[key, :].zero_()
                self.weights_down[key, value_lst[0]] = 1
        
        if len(self.fix_arch_normal_index.keys()) > 0:
            for key, value_lst in self.fix_arch_normal_index.items():
                self.weights_normal[key, :].zero_()
                self.weights_normal[key, value_lst[0]] = 1
        
        if len(self.fix_arch_up_index.keys()) > 0:
            for key, value_lst in self.fix_arch_up_index.items():
                self.weights_up[key, :].zero_()
                self.weights_up[key, value_lst[0]] = 1
    
    def _compute_alpha_loss(self):
        """Compute loss for architecture parameters"""
        loss_alpha_normal = torch.log((self.weights_normal * F.softmax(self.alphas_normal, dim=-1)).sum(-1)).sum()
        loss_alpha_down = torch.log((self.weights_down * F.softmax(self.alphas_down, dim=-1)).sum(-1)).sum()
        loss_alpha_up = torch.log((self.weights_up * F.softmax(self.alphas_up, dim=-1)).sum(-1)).sum()
        loss_alpha_network = torch.log((self.network_weight * F.softmax(self.network_weight, dim=-1)).sum(-1)).sum()
        loss_alpha_transformer = torch.log((self.transformer_weights * F.softmax(self.alphas_transformer_connections, dim=-1)).sum(-1)).sum()
        
        self.weights_normal.requires_grad_()
        self.weights_up.requires_grad_()
        self.weights_down.requires_grad_()
        self.network_weight.requires_grad_()
        self.transformer_weights.requires_grad_()
        
        return (loss_alpha_normal + loss_alpha_up + loss_alpha_down + loss_alpha_network + loss_alpha_transformer).sum()
    
    def _compute_architecture_gradients(self, preds, target, criterion, loss_alpha):
        """Compute gradients for architecture parameters"""
        preds = [pred.view(pred.size(0), -1) for pred in preds]
        target = target.view(target.size(0), -1)
        
        # Compute task loss
        error_loss = sum(criterion(pred, target) for pred in preds)
        
        # Initialize gradients
        self.weights_normal.grad = torch.zeros_like(self.weights_normal)
        self.weights_up.grad = torch.zeros_like(self.weights_up)
        self.weights_down.grad = torch.zeros_like(self.weights_down)
        self.network_weight.grad = torch.zeros_like(self.network_weight)
        self.transformer_weights.grad = torch.zeros_like(self.transformer_weights)
        
        # Backward pass
        (error_loss + loss_alpha).backward()
        
        # Compute rewards
        self.cell_up_reward = self.weights_up.grad.data.sum(dim=1)
        self.cell_down_reward = self.weights_down.grad.data.sum(dim=1)
        self.cell_normal_reward = self.weights_normal.grad.data.sum(dim=1)
        self.transformer_reward = self.transformer_weights.grad.data.sum(dim=1)
        
        # Apply rewards to alphas
        self.alphas_normal.grad.data.mul_(self.cell_normal_reward.view(-1, 1))
        self.alphas_down.grad.data.mul_(self.cell_down_reward.view(-1, 1))
        self.alphas_up.grad.data.mul_(self.cell_up_reward.view(-1, 1))
        self.alphas_transformer_connections.grad.data.mul_(self.transformer_reward.view(-1, 1))
    
    def _get_weights(self, log_alpha):
        """Sample weights from architecture parameters"""
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=F.softmax(log_alpha, dim=-1))
        return m.sample()
    
    def genotype(self):
        """Extract final architecture"""
        weight_normal = F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
        weight_down = F.softmax(self.alphas_down, dim=-1).data.cpu().numpy()
        weight_up = F.softmax(self.alphas_up, dim=-1).data.cpu().numpy()
        weight_transformer = F.softmax(self.alphas_transformer_connections, dim=-1).data.cpu().numpy()
        
        # Get transformer connections (those with "on" probability > 0.5)
        transformer_connections = []
        for i in range(self.num_transformer_connections):
            if weight_transformer[i, 1] > 0.5:  # Index 1 is "on"
                transformer_connections.append(i)
        
        # Parse cell structures
        normal_down_gen = self.normal_downup_parser(weight_normal.copy(), weight_down.copy(), 
                                                    self.CellLinkDownPos, self.CellPos,
                                                    self.switches_normal, self.switches_down, 
                                                    self.meta_node_num)
        normal_up_gen = self.normal_downup_parser(weight_normal.copy(), weight_up.copy(), 
                                                  self.CellLinkUpPos, self.CellPos,
                                                  self.switches_normal, self.switches_up, 
                                                  self.meta_node_num)
        normal_normal_gen = self.parser_normal_old(weight_normal.copy(), self.switches_normal, 
                                                   self.CellPos, self.meta_node_num)
        
        concat = range(2, self.meta_node_num + 2)
        geno_type = Genotype(
            normal_down=normal_down_gen, normal_down_concat=concat,
            normal_up=normal_up_gen, normal_up_concat=concat,
            normal_normal=normal_normal_gen, normal_normal_concat=concat,
            transformer_connections=transformer_connections,
        )
        return geno_type
    
    def normal_downup_parser(self, weight_normal, weight_down, CellLinkDownPos, CellPos, 
                            switches_normal, switches_down, meta_node_name):
        """Parse normal-down/up cell structure"""
        normalize_scale_nd = min(len(weight_normal[0]), len(weight_down[0])) / max(len(weight_normal[0]), len(weight_down[0]))
        down_normalize = True if len(weight_down[0]) < len(weight_normal[0]) else False
        normal_down_res = []
        
        for i in range(len(weight_normal)):
            if i in [1, 3, 6, 10]:
                if down_normalize:
                    temp_weight = (weight_down[i] * normalize_scale_nd + weight_normal[i]) / 2
                else:
                    temp_weight = (weight_down[i] + weight_normal[i] * normalize_scale_nd) / 2
            else:
                temp_weight = weight_normal[i]
            
            none_index = CellPos.index("none")
            temp_weight[switches_normal[i].index(1) if 1 in switches_normal[i] else none_index] = 0
            max_value = max(temp_weight)
            max_index = list(temp_weight).index(max_value)
            normal_down_res.append((max_value, CellPos[max_index]))
        
        # Keep top 2 edges per node
        n = 2
        start = 0
        normal_down_gen = []
        for i in range(meta_node_name):
            end = start + n
            node_edges = normal_down_res[start:end].copy()
            keep_edges = sorted(range(2 + i), key=lambda x: -node_edges[x][0])[:2]
            for j in keep_edges:
                normal_down_gen.append((node_edges[j][1], j))
            start = end
            n += 1
        
        return normal_down_gen
    
    def parser_normal_old(self, weights_normal, switches_normal, PRIMITIVES, meta_node_num=4):
        """Parse normal cell structure"""
        num_mixops = len(weights_normal)
        edge_keep = []
        
        for i in range(num_mixops):
            keep_obs = []
            none_index = PRIMITIVES.index("none")
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    keep_obs.append(j)
            
            max_value = max(weights_normal[i][k] for k in keep_obs if k != none_index)
            max_index = [k for k in keep_obs if weights_normal[i][k] == max_value][0]
            edge_keep.append((max_value, PRIMITIVES[max_index]))
        
        # Keep top 2 edges per node
        start = 0
        n = 2
        keep_operations = []
        for i in range(meta_node_num):
            end = start + n
            node_values = edge_keep[start:end].copy()
            keep_edges = sorted(range(2 + i), key=lambda x: -node_values[x][0])[:2]
            for j in keep_edges:
                keep_operations.append((node_values[j][1], j))
            start = end
            n += 1
        
        return keep_operations
    
    def load_alphas(self, alphas_dict):
        """Load architecture parameters from checkpoint"""
        self.alphas_down = alphas_dict['alphas_down']
        self.alphas_up = alphas_dict['alphas_up']
        self.alphas_normal = alphas_dict['alphas_normal']
        self.alphas_network = alphas_dict['alphas_network']
        if 'alphas_transformer_connections' in alphas_dict:
            self.alphas_transformer_connections = alphas_dict['alphas_transformer_connections']
        self._arch_parameters = [
            self.alphas_down,
            self.alphas_up,
            self.alphas_normal,
            self.alphas_network,
            self.alphas_transformer_connections,
        ]
    
    def alphas_dict(self):
        """Return architecture parameters as dictionary"""
        return {
            'alphas_down': self.alphas_down,
            'alphas_normal': self.alphas_normal,
            'alphas_up': self.alphas_up,
            'alphas_network': self.alphas_network,
            'alphas_transformer_connections': self.alphas_transformer_connections,
        }
    
    def arch_parameters(self):
        """Return architecture parameters"""
        return self._arch_parameters
    
    def weight_parameters(self):
        """Return weight parameters (non-architecture)"""
        return [param for name, param in self.named_parameters() if "alphas" not in name]
