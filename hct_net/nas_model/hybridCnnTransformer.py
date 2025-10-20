import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import math
import torch.nn.init as init
from torch.autograd import Variable
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hct_net.cell import Cell
from hct_net.genotypes import *
from hct_net.operations import *
from hct_net.nas_model.Kp_Trans.position_encoding import build_position_encoding
from hct_net.nas_model.Kp_Trans.DeformableTrans import DeformableTransformer


class hybridCnnTrans(nn.Module):
    def __init__(self, input_c=3, c=16, num_classes=1, meta_node_num=4, layers=7, dp=0,
                 use_sharing=True, double_down_channel=True, use_softmax_head=False,
                 switches_normal=[], switches_down=[], switches_up=[], switches_transformer=[],
                 early_fix_arch=False, gen_max_child_flag=False, random_sample=False):
        super(hybridCnnTrans, self).__init__()
        self.CellLinkDownPos = CellLinkDownPos
        self.CellPos = CellPos
        self.CellLinkUpPos = CellLinkUpPos
        self.switches_normal = switches_normal
        self.switches_down = switches_down
        self.switches_up = switches_up
        self.switches_transformer = switches_transformer
        self.dropout_prob = dp  # 0
        self.input_c = input_c  # 3
        self.num_class = num_classes  # 1
        self.meta_node_num = meta_node_num  # 4
        self.layers = layers  # 7
        self.use_sharing = use_sharing  # true
        self.double_down_channel = double_down_channel  # True
        self.use_softmax_head = use_softmax_head  # false
        self.depth = (self.layers + 1) // 2
        self.c_prev_prev = 32
        self.c_prev = 64
        self.early_fix_arch = early_fix_arch
        self.gen_max_child_flag = gen_max_child_flag
        self.random_sample = random_sample
        # print("swich_normal:", len(switches_up))
        # 3-->32
        self.stem0 = ConvOps(input_c, self.c_prev_prev, kernel_size=3, stride=1, ops_order='weight_norm_act')
        # 32-->64
        self.stem1 = ConvOps(self.c_prev_prev, self.c_prev, kernel_size=3, stride=2, ops_order='weight_norm_act')


        # transformer
        self.position_embed = build_position_encoding(mode='v2', hidden_dim=64)
        self.encoder_Detrans = DeformableTransformer(d_model=64, dim_feedforward=256, dropout=0.1, activation='gelu',
                                                     num_feature_levels=1, nhead=4, num_encoder_layers=6,
                                                     enc_n_points=4)

        # print("stem0:",self.stem0)


        # print("stem1:", self.stem1)

        # 1-->8 layer V cells node
        # steps, multiplier, C_prev_prev, C_prev, C, rate
        init_channel = c  # 16
        if self.double_down_channel:
            self.layers_channel = [self.meta_node_num * init_channel * pow(2, i) for i in
                                   range(self.depth)]  # 64 128 256 512
            self.cell_channels = [init_channel * pow(2, i) for i in range(self.depth)]  # 16 32 64 128
        else:
            self.layers_channel = [self.meta_node_num * init_channel for i in range(0, self.depth)]
            self.cell_channels = [init_channel for i in range(0, self.depth)]

        # print("2--",self.layers_channel,self.cell_channels)

        for i in range(1, self.layers):
            if i == 1:
                self.cell_1_1 = Cell(self.meta_node_num,
                                     -1, self.c_prev, self.cell_channels[1],
                                     switch_normal=self.switches_normal, switch_down=self.switches_down,
                                     switch_up=self.switches_up,
                                     cell_type="normal_down", dp=self.dropout_prob)
                # transformer
                self.position_embed = build_position_encoding(mode='v2', hidden_dim=128)
                self.encoder_Detrans = DeformableTransformer(d_model=128, dim_feedforward=512, dropout=0.1,
                                                             activation='gelu', num_feature_levels=1, nhead=4,
                                                             num_encoder_layers=6, enc_n_points=4)

                # print("cell_1_1",self.cell_1_1)
            elif i == 2:
                self.cell_2_0_0 = Cell(self.meta_node_num,
                                       -1, self.c_prev, self.cell_channels[0],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_normal", dp=self.dropout_prob)
                self.cell_2_0_1 = Cell(self.meta_node_num,
                                       self.c_prev, self.layers_channel[1], self.cell_channels[0],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_up", dp=self.dropout_prob)

                # Only create cell_2_2 if we have enough depth
                if len(self.cell_channels) > 2:
                    self.cell_2_2 = Cell(self.meta_node_num,
                                         -1, self.layers_channel[1], self.cell_channels[2],
                                         switch_normal=self.switches_normal, switch_down=self.switches_down,
                                         switch_up=self.switches_up,
                                         cell_type="normal_down", dp=self.dropout_prob)
                    # transformer
                    self.position_embed = build_position_encoding(mode='v2', hidden_dim=256)
                    self.encoder_Detrans = DeformableTransformer(d_model=256, dim_feedforward=1024, dropout=0.1, activation='gelu', num_feature_levels=1, nhead=4, num_encoder_layers=6, enc_n_points=4)


            elif i == 3:
                self.cell_3_1_0 = Cell(self.meta_node_num,
                                       self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_down", dp=self.dropout_prob)

                self.cell_3_1_1 = Cell(self.meta_node_num,
                                       -1, self.layers_channel[1], self.cell_channels[1],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_normal", dp=self.dropout_prob)

                self.cell_3_1_2 = Cell(self.meta_node_num,
                                       self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_3_3 = Cell(self.meta_node_num,
                                     -1, self.layers_channel[2], self.cell_channels[3],
                                     switch_normal=self.switches_normal, switch_down=self.switches_down,
                                     switch_up=self.switches_up,
                                     cell_type="normal_down", dp=self.dropout_prob)

            elif i == 4:
                self.cell_4_0_0 = Cell(self.meta_node_num,
                                       self.c_prev, self.layers_channel[0], self.cell_channels[0],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_normal", dp=self.dropout_prob)
                self.cell_4_0_1 = Cell(self.meta_node_num,
                                       self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_up", dp=self.dropout_prob)

                self.cell_4_2_0 = Cell(self.meta_node_num,
                                       self.layers_channel[2], self.layers_channel[1], self.cell_channels[2],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_down", dp=self.dropout_prob)
                self.cell_4_2_1 = Cell(self.meta_node_num,
                                       -1, self.layers_channel[2], self.cell_channels[2],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_normal", dp=self.dropout_prob)
                self.cell_4_2_2 = Cell(self.meta_node_num,
                                       self.layers_channel[2], self.layers_channel[3], self.cell_channels[2],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_up", dp=self.dropout_prob)


            elif i == 5:

                self.cell_5_1_0 = Cell(self.meta_node_num,
                                       self.layers_channel[1], self.layers_channel[0], self.cell_channels[1],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_down", dp=self.dropout_prob)
                self.cell_5_1_1 = Cell(self.meta_node_num,
                                       self.layers_channel[1], self.layers_channel[1], self.cell_channels[1],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_normal", dp=self.dropout_prob)
                self.cell_5_1_2 = Cell(self.meta_node_num,
                                       self.layers_channel[1], self.layers_channel[2], self.cell_channels[1],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_up", dp=self.dropout_prob)


            elif i == 6:
                self.cell_6_0_0 = Cell(self.meta_node_num,
                                       self.layers_channel[0], self.layers_channel[0], self.cell_channels[0],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_normal", dp=self.dropout_prob)
                self.cell_6_0_1 = Cell(self.meta_node_num,
                                       self.layers_channel[0], self.layers_channel[1], self.cell_channels[0],
                                       switch_normal=self.switches_normal, switch_down=self.switches_down,
                                       switch_up=self.switches_up,
                                       cell_type="normal_up", dp=self.dropout_prob)

        self.position_embed = build_position_encoding(mode='v2', hidden_dim=384)
        self.encoder_Detrans = DeformableTransformer(d_model=256, dim_feedforward=1024, dropout=0.1, activation='gelu', num_feature_levels=2, nhead=4, num_encoder_layers=6, enc_n_points=4)



        # transformer block
        self.transformer_blocks = nn.ModuleList()
        blocks = nn.ModuleList()

        self.position_embed_0 = build_position_encoding(mode='v2', hidden_dim=64)
        blocks.append(self.position_embed_0)
        self.transformer_blocks += [blocks]
        self.encoder_Detrans_0 = DeformableTransformer(d_model=64, dim_feedforward=256, dropout=0.1, activation='gelu',
                                                     num_feature_levels=1, nhead=4, num_encoder_layers=6,
                                                     enc_n_points=4)
        blocks.append(self.encoder_Detrans_0)
        self.transformer_blocks += [blocks]

        # transformer
        self.position_embed_1 = build_position_encoding(mode='v2', hidden_dim=128)
        blocks.append(self.position_embed_1)
        self.transformer_blocks += [blocks]
        self.encoder_Detrans_1 = DeformableTransformer(d_model=128, dim_feedforward=512, dropout=0.1,
                                                     activation='gelu', num_feature_levels=1, nhead=4,
                                                     num_encoder_layers=6, enc_n_points=4)
        blocks.append(self.encoder_Detrans_1)
        self.transformer_blocks += [blocks]

        # transformer
        self.position_embed_2 = build_position_encoding(mode='v2', hidden_dim=256)
        blocks.append(self.position_embed_2)
        self.transformer_blocks += [blocks]
        self.encoder_Detrans_2 = DeformableTransformer(d_model=256, dim_feedforward=1024, dropout=0.1, activation='gelu', num_feature_levels=1, nhead=4, num_encoder_layers=6, enc_n_points=4)
        blocks.append(self.encoder_Detrans_2)
        self.transformer_blocks += [blocks]

        print("self.transformer_blocks",self.transformer_blocks)

        self.cell_2_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')
        self.cell_4_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')
        self.cell_6_0_output = ConvOps(self.layers_channel[0], num_classes, kernel_size=1, dropout_rate=0.1,
                                       ops_order='weight')

        if self.use_softmax_head:
            self.softmax = nn.LogSoftmax(dim=1)

        self._init_arch_parameters()
        # self._init_weight_parameters()

        if self.early_fix_arch:  # true
            self.fix_arch_down_index = {}

        if self.early_fix_arch:  # true
            self.fix_arch_normal_index = {}

        if self.early_fix_arch:  # true
            self.fix_arch_up_index = {}
            
        if self.early_fix_arch:  # true
            self.fix_arch_transformer_index = {}
        
        # Cache for dynamic transformers
        self._transformer_cache = {}

    def _init_arch_parameters(self):
        normal_num_ops = np.count_nonzero(self.switches_normal[0])  # 7
        down_num_ops = np.count_nonzero(self.switches_down[0])  # 6
        up_num_ops = np.count_nonzero(self.switches_up[0])  # 4
        transformer_num_configs = np.count_nonzero(self.switches_transformer[0])  # 7 transformer layer configs

        # print("number",normal_num_ops,down_num_ops,up_num_ops, transformer_num_configs)

        k = sum(1 for i in range(self.meta_node_num) for n in range(2 + i))  # total number of input node 14

        #################################### 操作参数的初始化 ##########################################

        # self.alphas_down = nn.Parameter(torch.zeros(k, down_num_ops).normal_(1, 0.01))  # 生成正态分布  14*7的矩阵表示
        # self.weights_down = Variable(torch.zeros_like(self.alphas_down))
        # self.alphas_normal = nn.Parameter(torch.zeros(k, normal_num_ops).normal_(1, 0.01))
        # self.weights_normal = Variable(torch.zeros_like(self.alphas_normal))
        # self.alphas_up = nn.Parameter(torch.zeros(k, up_num_ops).normal_(1, 0.01))
        # self.weights_up = Variable(torch.zeros_like(self.alphas_up))

        self.alphas_down = nn.Parameter(1e-3 * torch.randn(k, down_num_ops))  # 生成正态分布  14*7的矩阵表示
        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, normal_num_ops))
        self.alphas_up = nn.Parameter(1e-3 * torch.randn(k, up_num_ops))

        #################################### 架构参数的初始化 ##########################################
        self.alphas_network = nn.Parameter(1e-3 * torch.randn(self.layers, self.depth, 3))  # 这是autodeeplab上每个点 存在的每个cell的概率
        # Transformer layer configuration alphas - each layer depth has choices of num_layers
        self.alphas_transformer = nn.Parameter(1e-3 * torch.randn(self.layers, transformer_num_configs))
        
        # Transformer connection alphas - search which connections to enable/disable
        # Dynamic based on number of layers
        from hct_net.genotypes import get_transformer_connection_pairs
        self.transformer_connection_pairs = get_transformer_connection_pairs(self.layers)
        self.num_transformer_connections = len(self.transformer_connection_pairs)
        if self.num_transformer_connections > 0:
            self.alphas_transformer_connections = nn.Parameter(1e-3 * torch.randn(self.num_transformer_connections, 2))  # on/off for each connection
        else:
            self.alphas_transformer_connections = None  # No connections for very shallow networks
        # self.alphas_network = nn.Parameter(torch.zeros(self.layers, self.depth, 3).normal_(1, 0.01))
        # self.network_weight = Variable(torch.zeros_like(self.alphas_network))
        # print(" self.alphas_network:",self.alphas_network)

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():  # 函数model.named_parameters()，返回各层中参数名称和数据。
            if 'alphas' in n:  # it is a trick, because the parameter name is the prefix of self.alphas_xxx
                self._alphas.append((n, p))
        # print("self._alpha:",self._alpha)

        self._arch_parameters = [
            # cell
            self.alphas_down,
            self.alphas_up,
            self.alphas_normal,
            # network
            self.alphas_network,
            # transformer
            self.alphas_transformer,
        ]
        # Add transformer connections if available
        if self.alphas_transformer_connections is not None:
            self._arch_parameters.append(self.alphas_transformer_connections)

    def _init_weight_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
    
    def compute_architecture_complexity(self):
        """
        Calculate architecture complexity based on:
        1. Number of active operations in CNN cells
        2. Number of transformer layers
        3. Number of active transformer connections
        """
        complexity = 0.0
        
        # 1. CNN operations complexity
        # Weight by operation cost (relative computational cost)
        operation_costs = {
            'none': 0.0,
            'identity': 0.1,
            'avg_pool': 0.2,
            'max_pool': 0.2,
            'conv': 1.0,
            'dep_conv': 0.6,
            'dil_conv': 1.2,
            'shuffle_conv': 0.8,
            'cweight': 0.7,
            'Dynamic_convOPS': 1.5,
            'down_conv': 1.0,
            'down_dep_conv': 0.6,
            'down_dil_conv': 1.2,
            'down_cweight': 0.7,
            'down_Dynamic_convOPS': 1.5,
            'up_conv': 1.0,
            'up_dep_conv': 0.6,
            'up_dil_conv': 1.2,
            'up_cweight': 0.7,
            'up_Dynamic_convOPS': 1.5,
        }
        
        # Get operation probabilities
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_down = F.softmax(self.alphas_down, dim=-1)
        weights_up = F.softmax(self.alphas_up, dim=-1)
        
        # Calculate expected complexity for each mixop
        for i in range(len(self.alphas_normal)):
            for j, active in enumerate(self.switches_normal[i]):
                if active:
                    op_name = self.CellPos[j]
                    cost = operation_costs.get(op_name, 1.0)
                    complexity += weights_normal[i, j] * cost
        
        for i in range(len(self.alphas_down)):
            for j, active in enumerate(self.switches_down[i]):
                if active:
                    op_name = self.CellLinkDownPos[j]
                    cost = operation_costs.get(op_name, 1.0)
                    complexity += weights_down[i, j] * cost
        
        for i in range(len(self.alphas_up)):
            for j, active in enumerate(self.switches_up[i]):
                if active:
                    op_name = self.CellLinkUpPos[j]
                    cost = operation_costs.get(op_name, 1.0)
                    complexity += weights_up[i, j] * cost
        
        # 2. Transformer layers complexity
        # More layers = higher complexity
        from hct_net.genotypes import TransformerLayerConfigs
        if hasattr(self, 'alphas_transformer') and self.alphas_transformer is not None:
            weights_transformer = F.softmax(self.alphas_transformer, dim=-1)
            for layer_idx in range(self.layers):
                expected_num_layers = 0.0
                for config_idx, num_layers in enumerate(TransformerLayerConfigs):
                    # Guard indexing if shapes don't match (robust for smaller models)
                    if layer_idx < weights_transformer.shape[0] and config_idx < weights_transformer.shape[1]:
                        expected_num_layers += weights_transformer[layer_idx, config_idx] * num_layers
                # Transformer is more expensive than CNN
                complexity += expected_num_layers * 2.0  # Each transformer layer costs 2x a conv layer
        else:
            # No transformer alphas configured -> assume zero transformer layers
            pass

        # 3. Transformer connections complexity
        # Each active connection adds overhead
        if hasattr(self, 'alphas_transformer_connections') and self.alphas_transformer_connections is not None:
            weights_transformer_conn = F.softmax(self.alphas_transformer_connections, dim=-1)
            # Ensure num_transformer_connections exists
            num_conns = getattr(self, 'num_transformer_connections', weights_transformer_conn.shape[0])
            for conn_idx in range(num_conns):
                # Index 1 = connection ON, Index 0 = connection OFF
                if conn_idx < weights_transformer_conn.shape[0] and weights_transformer_conn.shape[1] > 1:
                    prob_on = weights_transformer_conn[conn_idx, 1]
                else:
                    prob_on = 0.0
                complexity += prob_on * 1.5  # Each connection adds moderate cost
        else:
            # No transformer connections configured -> no extra complexity
            pass
        
        return complexity

    def _get_dynamic_transformer(self, d_model, dim_feedforward, layer_idx):
        """
        Dynamically create transformer based on architecture search parameters
        """
        if not hasattr(self, 'alphas_transformer'):
            # Fallback to original fixed layers if alphas not initialized
            num_layers = 6
        else:
            # Get number of layers from alpha weights
            if self.gen_max_child_flag:
                # Use argmax for child generation
                layer_config_idx = torch.argmax(self.alphas_transformer[layer_idx]).item()
            else:
                # Use weighted sampling during search
                layer_probs = F.softmax(self.alphas_transformer[layer_idx], dim=-1)
                if self.training:
                    # Sample during training
                    layer_config_idx = torch.multinomial(layer_probs, 1).item()
                else:
                    # Use argmax during inference
                    layer_config_idx = torch.argmax(layer_probs).item()
            
            # Map to actual number of layers
            from hct_net.genotypes import TransformerLayerConfigs
            num_layers = TransformerLayerConfigs[layer_config_idx]
        
        # Create cache key
        cache_key = (layer_idx, d_model, dim_feedforward, num_layers)
        
        # Check cache first
        if hasattr(self, '_transformer_cache') and cache_key in self._transformer_cache:
            return self._transformer_cache[cache_key]
        
        # Create transformer with dynamic number of layers
        transformer = DeformableTransformer(
            d_model=d_model, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, 
            activation='gelu',
            num_feature_levels=1, 
            nhead=4, 
            num_encoder_layers=num_layers,
            enc_n_points=4
        )
        
        # Move transformer to the same device as the model
        device = next(self.parameters()).device
        transformer = transformer.to(device)
        
        # Cache the transformer
        if not hasattr(self, '_transformer_cache'):
            self._transformer_cache = {}
        self._transformer_cache[cache_key] = transformer
        
        return transformer

    def posi_mask(self, x):
        #print("###x####", x.shape)
        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            print("######lvl####",lvl)
            print("fea",fea.shape)
            print("######fea.shape[0]######",fea.shape[0])
            print("######fea.shape[1]######",fea.shape[1])
            print("######fea.shape[2]######",fea.shape[2])
            #print("######fea.shape[3]######",fea.shape[3])

            if lvl > 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros((fea.shape[0],fea.shape[1], fea.shape[2]), dtype=torch.bool, device=fea.device))
        x_fea.append(x)
        x_posemb.append(self.position_embed(x))
        masks.append(torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device))
        # print("x_posemb",len(x_posemb))
        return x_fea, masks, x_posemb

    def forward(self, input, target, criterion):
        '''
        :param input:
        :return:
        '''
        _, _, h, w = input.size()  # 3*256*256
        '''
        alphas_normal=self.alphas_normal
        alphas_down=self.alphas_down
        alphas_up=self.alphas_up
        '''
        if self.gen_max_child_flag:
            self.weights = torch.zeros_like(self.log_alpha).scatter_(1,
                                                                     torch.argmax(self.log_alpha, dim=-1).view(-1, 1),
                                                                     1)
            self.weights_normal = torch.zeros_like(self.alphas_normal).scatter_(1, torch.argmax(self.alphas_normal,
                                                                                                dim=-1).view(-1, 1), 1)
            self.weights_down = torch.zeros_like(self.alphas_down).scatter_(1,
                                                                            torch.argmax(self.alphas_down, dim=-1).view(
                                                                                -1, 1), 1)
            self.weights_up = torch.zeros_like(self.alphas_up).scatter_(1, torch.argmax(self.alphas_up, dim=-1).view(-1,
                                                                                                                     1),
                                                                        1)
            self.network_weight = F.softmax(self.alphas_network, dim=-1)  # beta
            # Transformer connections weight
            self.weights_transformer_conn = torch.zeros_like(self.alphas_transformer_connections).scatter_(1, 
                                                                            torch.argmax(self.alphas_transformer_connections, dim=-1).view(-1, 1), 1)
        else:
            self.weights_normal = self._get_weights(self.alphas_normal)
            self.weights_down = self._get_weights(self.alphas_down)
            self.weights_up = self._get_weights(self.alphas_up)

            self.network_weight = F.softmax(self.alphas_network, dim=-1)  # beta
            # Transformer connections weight - for each connection: [prob_off, prob_on]
            if self.alphas_transformer_connections is not None:
                self.weights_transformer_conn = self._get_weights(self.alphas_transformer_connections)
            else:
                self.weights_transformer_conn = None

        '''print("self.weights_normal",self.weights_normal)
        print("self.weights_down", self.weights_down)
        print("self.weights_up", self.weights_up)'''

        if self.early_fix_arch:
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
                    
            # Fix transformer layers during search
            if hasattr(self, 'fix_arch_transformer_index') and len(self.fix_arch_transformer_index.keys()) > 0:
                for key, value_lst in self.fix_arch_transformer_index.items():
                    # Store fixed transformer configuration for this layer
                    pass  # Will be handled in _get_dynamic_transformer

        if not self.random_sample and not self.gen_max_child_flag:
            cate_prob_normal = F.softmax(self.alphas_normal, dim=-1)
            cate_prob_down = F.softmax(self.alphas_down, dim=-1)
            cate_prob_up = F.softmax(self.alphas_up, dim=-1)
            cate_prob_network = F.softmax(self.alphas_network, dim=-1)

            self.cate_prob_normal = cate_prob_normal.clone().detach()
            self.cate_prob_down = cate_prob_down.clone().detach()
            self.cate_prob_up = cate_prob_up.clone().detach()
            self.cate_prob_network = cate_prob_network.clone().detach()

            loss_alpha_normal = torch.log((self.weights_normal * F.softmax(self.alphas_normal, dim=-1)).sum(-1)).sum()
            loss_alpha_down = torch.log((self.weights_down * F.softmax(self.alphas_down, dim=-1)).sum(-1)).sum()
            loss_alpha_up = torch.log((self.weights_up * F.softmax(self.alphas_up, dim=-1)).sum(-1)).sum()
            loss_alpha_network = torch.log((self.network_weight * F.softmax(self.network_weight, dim=-1)).sum(-1)).sum()
            self.weights_normal.requires_grad_()
            self.weights_up.requires_grad_()
            self.weights_down.requires_grad_()
            self.network_weight.requires_grad_()
            # print("normal,up,down",loss_alpha_normal,loss_alpha_up,loss_alpha_down)
            loss_alpha = (loss_alpha_normal + loss_alpha_up + loss_alpha_down + loss_alpha_network).sum()
            # print("sum(loss_alpha):",loss_alpha)
        # layer 0
        self.stem0_f = self.stem0(input)
        self.stem1_f = self.stem1(self.stem0_f)
        # print("####self.stem1_f#####",self.stem1_f.shape)

        #################################################### transformer ########################################3
        ###############################position##############################
        x_fea_0 = []
        x_posemb_0 = []
        masks_0 = []
        x_fea_0.append(self.stem1_f)
        x_posemb_0.append(self.position_embed_0(self.stem1_f))
        masks_0.append(torch.zeros((self.stem1_f.shape[0], self.stem1_f.shape[2], self.stem1_f.shape[3]), dtype=torch.bool, device=self.stem1_f.device))
        # Dynamic transformer for layer 0
        dynamic_transformer_0 = self._get_dynamic_transformer(d_model=64, dim_feedforward=256, layer_idx=0)
        x_trans_0 = dynamic_transformer_0(x_fea_0, masks_0, x_posemb_0)  # [4,16384,64]

        self.cell_0 = x_trans_0[:, :1048576, :].transpose(-1, -2).view(self.stem1_f.shape)  # [4,64,128,128]

        # layer 1
        self.cell_1_1_f = self.cell_1_1(None, self.stem1_f, self.weights_normal, self.weights_down, self.weights_up)
        # print("####self.cell_1_1_f#####",self.cell_1_1_f.shape)

        #################################################### transformer ########################################3
        ###############################position##############################
        x_fea_1 = []
        x_posemb_1 = []
        masks_1 = []
        x_fea_1.append(self.cell_1_1_f)
        x_posemb_1.append(self.position_embed_1(self.cell_1_1_f))
        masks_1.append(torch.zeros((self.cell_1_1_f.shape[0], self.cell_1_1_f.shape[2], self.cell_1_1_f.shape[3]), dtype=torch.bool, device=self.cell_1_1_f.device))
        # Dynamic transformer for layer 1
        dynamic_transformer_1 = self._get_dynamic_transformer(d_model=128, dim_feedforward=512, layer_idx=1)
        x_trans_1 = dynamic_transformer_1(x_fea_1, masks_1, x_posemb_1)  # [4,16384,64]
        # print("####x_trans_1#####",x_trans_1.shape)
        self.cell_1 = x_trans_1[:, :524288, :].transpose(-1, -2).view(self.cell_1_1_f.shape)  # [4,64,128,128]
        # print("###self.cell_1##",self.cell_1.shape)

        # layer 2
        self.cell_2_0_f = self.cell_2_0_0(None, self.stem1_f, self.weights_normal, self.weights_down, self.weights_up) * \
                          self.network_weight[2][0][1] / (self.network_weight[2][0][1] + self.network_weight[2][0][2]) + \
                          self.cell_2_0_1(self.stem1_f, self.cell_1_1_f, self.weights_normal, self.weights_down,
                                          self.weights_up) * self.network_weight[2][0][2] / (
                                  self.network_weight[2][0][1] + self.network_weight[2][0][2])
        # print("####self.cell_2_0_f#####",self.cell_2_0_f.shape)
        
        # Only process deeper layers if they exist (UNet7+)
        if hasattr(self, 'cell_2_2'):
            self.cell_2_2_f = self.cell_2_2(None, self.cell_1_1_f, self.weights_normal, self.weights_down, self.weights_up)
            # print("####self.cell_2_2_f#####",self.cell_2_2_f.shape)

            #################################################### transformer ########################################3
            ###############################position##############################
            x_fea_2 = []
            x_posemb_2 = []
            masks_2 = []
            x_fea_2.append(self.cell_2_2_f)
            x_posemb_2.append(self.position_embed_2(self.cell_2_2_f))
            masks_2.append(torch.zeros((self.cell_2_2_f.shape[0], self.cell_2_2_f.shape[2],self.cell_2_2_f.shape[3]),dtype=torch.bool, device=self.cell_2_2_f.device))
            #print("*******masks*****",masks.shape)
            # Dynamic transformer for layer 2  
            dynamic_transformer_2 = self._get_dynamic_transformer(d_model=256, dim_feedforward=1024, layer_idx=2)
            x_trans_2 = dynamic_transformer_2(x_fea_2, masks_2, x_posemb_2) #[4,16384,64]
            #print("###x_trans_0####",x_trans_0.shape)
            #self.cell_0 = self.transposeconv_stage2(x_trans_0[:, :512, :].transpose(-1, -2).view(self.stem1_f.shape))
            if x_trans_2 is not None:
                self.cell_2 = x_trans_2[:, :262144, :].transpose(-1, -2).view(self.cell_2_2_f.shape) #[4,64,128,128]
            else:
                self.cell_2 = self.cell_2_2_f  # Fallback to original feature if transformer fails
            #print("###self.cell_2##",self.cell_2.shape)
        else:
            # For UNet3, no deeper encoder layers
            self.cell_2 = None

        # layer 3 - only for UNet7+
        if self.layers > 3 and hasattr(self, 'cell_3_1_0'):
            self.cell_3_1_f = self.cell_3_1_0(self.cell_1_1_f, self.cell_2_0_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[3][1][0] + \
                              self.cell_3_1_1(None, self.cell_1_1_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[3][1][1] + \
                              self.cell_3_1_2(self.cell_1_1_f, self.cell_2_2_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[3][1][2]

            self.cell_3_3_f = self.cell_3_3(None, self.cell_2_2_f, self.weights_normal, self.weights_down, self.weights_up)
            # + \self.cell_2_2
            # print("####self.cell_3_3_f####",self.cell_3_3_f.shape)
        
        # layer 4 - only for UNet7+
        if self.layers > 4 and hasattr(self, 'cell_4_0_0'):
            self.cell_4_0_f = self.cell_4_0_0(self.stem1_f, self.cell_2_0_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[4][0][1] / (
                                      self.network_weight[4][0][1] + self.network_weight[4][0][2]) + \
                              self.cell_4_0_1(self.cell_2_0_f, self.cell_3_1_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[4][0][2] / (
                                      self.network_weight[4][0][1] + self.network_weight[4][0][2])
            # print("####self.cell_3_3_f####",self.cell_3_3_f.shape)
            
            self.cell_4_2_f = self.cell_4_2_0(self.cell_2_2_f, self.cell_3_1_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[4][2][0] + \
                              self.cell_4_2_1(None, self.cell_2_2_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[4][2][1] + \
                              self.cell_4_2_2(self.cell_2_2_f, self.cell_3_3_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[4][2][2]
            # Note: Layer 4 doesn't receive transformer connection in formula (i, n-i+2)
            # print("####self.cell_4_2_f ###",self.cell_4_2_f.shape)
        
        # layer 5 - only for UNet7+
        if self.layers > 5 and hasattr(self, 'cell_5_1_0'):
            self.cell_5_1_f = self.cell_5_1_0(self.cell_3_1_f, self.cell_4_0_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[5][1][0] + \
                              self.cell_5_1_1(self.cell_1_1_f, self.cell_3_1_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[5][1][1] + \
                              self.cell_5_1_2(self.cell_3_1_f, self.cell_4_2_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[5][1][2]
            # print("####self.cell_5_1_f ###",self.cell_5_1_f.shape)
        
        # layer 6 - only for UNet7+
        if self.layers > 6 and hasattr(self, 'cell_6_0_0'):
            self.cell_6_0_f = self.cell_6_0_0(self.cell_2_0_f, self.cell_4_0_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[6][0][1] / (
                                      self.network_weight[6][0][1] + self.network_weight[6][0][2]) + \
                              self.cell_6_0_1(self.cell_4_0_f, self.cell_5_1_f, self.weights_normal, self.weights_down,
                                              self.weights_up) * self.network_weight[6][0][2] / (
                                      self.network_weight[6][0][1] + self.network_weight[6][0][2])

            # print("####self.cell_6_0_f ###",self.cell_6_0_f.shape)

        # Generate outputs based on available layers
        preds = []
        if hasattr(self, 'cell_2_0_output'):
            self.ouput_2_0 = self.cell_2_0_output(self.cell_2_0_f)
            self.ouput_2_0 = F.interpolate(self.ouput_2_0, size=(h, w), mode='bilinear', align_corners=False)
            preds.append(self.ouput_2_0)
        
        if self.layers > 4 and hasattr(self, 'cell_4_0_output') and hasattr(self, 'cell_4_0_f'):
            self.ouput_4_0 = self.cell_4_0_output(self.cell_4_0_f)
            self.ouput_4_0 = F.interpolate(self.ouput_4_0, size=(h, w), mode='bilinear', align_corners=False)
            preds.append(self.ouput_4_0)
        
        if self.layers > 6 and hasattr(self, 'cell_6_0_output') and hasattr(self, 'cell_6_0_f'):
            self.ouput_6_0 = self.cell_6_0_output(self.cell_6_0_f)
            self.ouput_6_0 = F.interpolate(self.ouput_6_0, size=(h, w), mode='bilinear', align_corners=False)
            preds.append(self.ouput_6_0)
        assert isinstance(preds, list)
        preds = [pred.view(pred.size(0), -1) for pred in preds]
        target = target.view(target.size(0), -1)
        if not self.random_sample and self.training and not self.gen_max_child_flag:
            # Compute task loss
            for i in range(len(preds)):
                if i == 0:
                    target1_loss = criterion(preds[i], target)
                target1_loss += criterion(preds[i], target)

            error_loss = target1_loss
            
            # Compute architecture complexity loss
            complexity = self.compute_architecture_complexity()
            complexity_weight = 0.001  # Adjustable weight for complexity regularization
            complexity_loss = complexity_weight * complexity
            
            # Total loss = task loss + complexity loss
            total_loss = error_loss + complexity_loss
            
            # print("error_loss",target1_loss, "complexity_loss", complexity_loss)
            self.weights_normal.grad = torch.zeros_like(self.weights_normal)
            # print("cell_up_reward", self.weights_normal.grad)
            self.weights_up.grad = torch.zeros_like(self.weights_up)
            # print("cell_up_reward", self.weights_up.grad)
            self.weights_down.grad = torch.zeros_like(self.weights_down)
            self.network_weight.grad = torch.zeros_like(self.network_weight)
            if self.weights_transformer_conn is not None:
                self.weights_transformer_conn.grad = torch.zeros_like(self.weights_transformer_conn)

            (total_loss + loss_alpha).backward()

            # print("self.weights_normal.grad",self.weights_down.grad)
            self.cell_up_reward = self.weights_up.grad.data.sum(dim=1)
            self.cell_down_reward = self.weights_down.grad.data.sum(dim=1)
            self.cell_normal_reward = self.weights_normal.grad.data.sum(dim=1)
            if self.weights_transformer_conn is not None:
                self.transformer_conn_reward = self.weights_transformer_conn.grad.data.sum(dim=1)
            # self.cell_network_reward = self.network_weight.grad.data.sum(dim=1)
            # print("up,normal,down", self.cell_up_reward,self.cell_normal_reward,self.cell_down_reward)

            self.alphas_normal.grad.data.mul_(self.cell_normal_reward.view(-1, 1))
            self.alphas_down.grad.data.mul_(self.cell_down_reward.view(-1, 1))
            self.alphas_up.grad.data.mul_(self.cell_up_reward.view(-1, 1))
            if self.alphas_transformer_connections is not None:
                self.alphas_transformer_connections.grad.data.mul_(self.transformer_conn_reward.view(-1, 1))
            # self.alphas_network.grad.data.mul_(self.cell_network_reward.view(-1, 1))

            # print("self.weights_normal.grad", self.alphas_down.grad)
            # print("self.weights_normal.grad", self.alphas_down)

        return preds

    def load_alphas(self, alphas_dict):
        self.alphas_down = alphas_dict['alphas_down']
        self.alphas_up = alphas_dict['alphas_up']
        self.alphas_normal = alphas_dict['alphas_normal']
        self.alphas_network = alphas_dict['alphas_network']
        if 'alphas_transformer' in alphas_dict:
            self.alphas_transformer = alphas_dict['alphas_transformer']
        if 'alphas_transformer_connections' in alphas_dict:
            self.alphas_transformer_connections = alphas_dict['alphas_transformer_connections']
        self._arch_parameters = [
            self.alphas_down,
            self.alphas_up,
            self.alphas_normal,
            self.alphas_network
        ]
        if hasattr(self, 'alphas_transformer'):
            self._arch_parameters.append(self.alphas_transformer)
        if hasattr(self, 'alphas_transformer_connections'):
            self._arch_parameters.append(self.alphas_transformer_connections)

    def alphas_dict(self):
        alphas_dict = {
            'alphas_down': self.alphas_down,
            'alphas_normal': self.alphas_normal,
            'alphas_up': self.alphas_up,
            'alphas_network': self.alphas_network
        }
        if hasattr(self, 'alphas_transformer'):
            alphas_dict['alphas_transformer'] = self.alphas_transformer
        if hasattr(self, 'alphas_transformer_connections'):
            alphas_dict['alphas_transformer_connections'] = self.alphas_transformer_connections
        return alphas_dict

    def arch_parameters(self):
        return self._arch_parameters

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "alphas" not in name]

    def decode_network(self):
        '''
        :return: Get the path probability and the largest set of paths
        '''
        # Get path weights
        network_parameters = F.softmax(self.arch_parameters()[1], dim=-1).data.cpu().numpy() * 10
        # Take only valid path branches

    def _get_weights(self, log_alpha):
        m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=F.softmax(log_alpha, dim=-1))
        return m.sample()
    
    def _get_connection_weight(self, encoder_layer, decoder_layer):
        """
        Get transformer connection weight for a specific encoder-decoder pair
        Returns 1.0 if connection not searchable, or the searched weight
        """
        if self.alphas_transformer_connections is None or self.num_transformer_connections == 0:
            return 1.0  # Default: all connections active
        
        # Find connection index
        try:
            conn_idx = self.transformer_connection_pairs.index((encoder_layer, decoder_layer))
            conn_weight = F.softmax(self.alphas_transformer_connections[conn_idx], dim=-1)[1]  # prob ON
            return conn_weight
        except (ValueError, IndexError):
            # Connection not in search space, default to ON
            return 1.0

    def genotype(self):
        '''
        :return: Get the structure of the cell
        '''
        weight_normal = F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy()
        weight_down = F.softmax(self.alphas_down, dim=-1).data.cpu().numpy()
        weight_up = F.softmax(self.alphas_up, dim=-1).data.cpu().numpy()
        num_mixops = len(weight_normal)
        assert num_mixops == 14
        assert len(self.switches_normal) == num_mixops and len(self.switches_down) == num_mixops and len(
            self.switches_up) == num_mixops
        # get the normal_down
        keep_num_normal = np.count_nonzero(self.switches_normal[0])
        keep_num_down = np.count_nonzero(self.switches_down[0])
        keep_num_up = np.count_nonzero(self.switches_up[0])
        assert keep_num_normal == len(weight_normal[0]) and keep_num_down == len(weight_down[0]) and keep_num_up == len(
            weight_up[0])

        normal_down_gen = self.normal_downup_parser(weight_normal.copy(), weight_down.copy(), self.CellLinkDownPos,
                                                    self.CellPos,
                                                    self.switches_normal, self.switches_down, self.meta_node_num)
        normal_up_gen = self.normal_downup_parser(weight_normal.copy(), weight_up.copy(), self.CellLinkUpPos,
                                                  self.CellPos,
                                                  self.switches_normal, self.switches_up, self.meta_node_num)
        normal_normal_gen = self.parser_normal_old(weight_normal.copy(), self.switches_normal, self.CellPos,
                                                   self.meta_node_num)

        # Extract transformer connections
        transformer_connections = []
        if hasattr(self, 'alphas_transformer_connections') and self.alphas_transformer_connections is not None:
            weight_transformer_conn = F.softmax(self.alphas_transformer_connections, dim=-1).data.cpu().numpy()
            for conn_idx, (enc_layer, dec_layer) in enumerate(self.transformer_connection_pairs):
                # Index 1 = ON, Index 0 = OFF
                prob_on = weight_transformer_conn[conn_idx, 1]
                prob_off = weight_transformer_conn[conn_idx, 0]
                is_enabled = prob_on > prob_off
                transformer_connections.append({
                    'encoder_layer': enc_layer,
                    'decoder_layer': dec_layer,
                    'enabled': bool(is_enabled),
                    'prob_on': float(prob_on)
                })

        concat = range(2, self.meta_node_num + 2)
        geno_type = Genotype(
            normal_down=normal_down_gen, normal_down_concat=concat,
            normal_up=normal_up_gen, normal_up_concat=concat,
            normal_normal=normal_normal_gen, normal_normal_concat=concat,
            transformer_connections=transformer_connections,
        )
        return geno_type

    def normal_downup_parser(self, weight_normal, weight_down, CellLinkDownPos, CellPos, switches_normal, switches_down,
                             meta_node_name):
        # get the normal_down
        normalize_sacle_nd = min(len(weight_normal[0]), len(weight_down[0])) / max(len(weight_normal[0]),
                                                                                   len(weight_down[0]))  # ?
        down_normalize = True if len(weight_down[0]) < len(weight_normal[0]) else False
        normal_down_res = []
        for i in range(len(weight_normal)):
            if i in [1, 3, 6, 10]:  # pre节点用于上下采样操作
                if down_normalize:
                    mixop_array = weight_down[i] * normalize_sacle_nd
                else:
                    mixop_array = weight_down[i]
                keep_ops_index = []
                for j in range(len(CellLinkDownPos)):
                    if switches_down[i][j]:
                        keep_ops_index.append(j)
                max_value, max_index = float(np.max(mixop_array)), int(np.argmax(mixop_array))
                max_index_pri = keep_ops_index[max_index]
                max_op_name = CellLinkDownPos[max_index_pri]
                assert max_op_name != 'none'
                normal_down_res.append((max_value, max_op_name))
            else:
                if down_normalize:
                    mixop_array = weight_normal[i]
                else:
                    mixop_array = weight_normal[i] * normalize_sacle_nd
                keep_ops_index = []
                for j in range(len(CellPos)):
                    if switches_normal[i][j]:
                        keep_ops_index.append(j)
                assert CellPos.index('none') == 0
                # Excluding none
                if switches_normal[i][0]:
                    mixop_array[0] = 0
                max_value, max_index = float(np.max(mixop_array)), int(np.argmax(mixop_array))
                max_index_pri = keep_ops_index[max_index]
                max_op_name = CellPos[max_index_pri]
                assert max_op_name != 'none'
                normal_down_res.append((max_value, max_op_name))
        # get the final cell genotype based in normal_down_res
        # print(normal_down_res)
        n = 2
        start = 0
        normal_down_gen = []
        for i in range(meta_node_name):
            end = start + n
            node_egdes = normal_down_res[start:end].copy()  # 节点所对应 的所有边
            keep_edges = sorted(range(2 + i), key=lambda x: -node_egdes[x][0])[:2]
            for j in keep_edges:
                op_name = node_egdes[j][1]
                normal_down_gen.append((op_name, j))
            start = end
            n += 1
        return normal_down_gen

    def parser_normal_old(self, weights_normal, siwtches_normal, PRIMITIVES, meta_node_num=4):
        num_mixops = len(weights_normal)
        assert len(siwtches_normal) == len(weights_normal), "The mixop num is not right !"
        num_operations = np.count_nonzero(siwtches_normal[0])
        for i in range(num_mixops):
            if siwtches_normal[i][0] == True:
                weights_normal[i][0] = 0
        edge_keep = []
        for i in range(num_mixops):
            keep_obs = []
            none_index = PRIMITIVES.index("none")
            for j in range(len(PRIMITIVES)):
                if siwtches_normal[i][j]:
                    keep_obs.append(j)
            # find max operation
            assert len(keep_obs) == num_operations, "The mixop {}`s keep ops is wrong !".format(i)
            # get the max op index and the max value apart from  zero
            max_value, max_index = float(np.max(weights_normal[i])), int(np.argmax(weights_normal[i]))
            max_index_pri = keep_obs[max_index]
            # print("i:{} cur:{} Pro:{} operation:{} max_value:{}".format(i,max_index,max_index_pri,PRIMITIVES[max_index],max_value))
            assert max_index_pri != none_index, "The none be choose !"
            edge_keep.append((max_value, PRIMITIVES[max_index_pri]))
        # keep two edge for every node
        start = 0
        n = 2
        keep_operations = []
        # 2,3,4,5
        for i in range(meta_node_num):
            end = start + n  # 0~1 2~4 5~8 9~14
            node_values = edge_keep[start:end].copy()
            # The edge num of the ith point is 2+i
            keep_edges = sorted(range(2 + i), key=lambda x: -node_values[x][0])[:2]
            for j in keep_edges:
                keep_op = node_values[j][1]
                keep_operations.append((keep_op, j))
            start = end
            n += 1
        # return
        return keep_operations


if __name__ == "__main__":

    normal_num_ops = len(CellPos)
    down_num_ops = len(CellLinkDownPos)
    up_num_ops = len(CellLinkUpPos)

    switches_normal = []
    switches_down = []
    switches_up = []
    for i in range(14):
        switches_normal.append([True for j in range(len(CellPos))])
    for i in range(14):
        switches_down.append([True for j in range(len(CellLinkDownPos))])
    for i in range(14):
        switches_up.append([True for j in range(len(CellLinkUpPos))])

    model = hybridCnnTrans(input_c=3, c=16, num_classes=1, meta_node_num=4, layers=7, dp=0,
                           use_sharing=True, double_down_channel=True, use_softmax_head=False,
                           switches_normal=switches_normal, switches_down=switches_down, switches_up=switches_up)

    x = torch.FloatTensor(torch.ones(1, 3, 256, 256))
    ress = model(x)
    for res in ress:
        print(res.size())
    arch_para = model.arch_parameters()
    print(model.genotype())