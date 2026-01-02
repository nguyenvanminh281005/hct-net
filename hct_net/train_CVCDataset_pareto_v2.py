import os
import time
import argparse
from tqdm import tqdm
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.utils.data as data
import torch.nn.functional as F
import wandb
import warnings

# Suppress pydantic warnings from wandb
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

from genotypes import CellLinkDownPos, CellLinkUpPos, CellPos
from nas_model import get_models

sys.path.append('../')
from datasets import get_dataloder, datasets_dict
from utils import save_checkpoint, calc_parameters_count, get_logger, get_gpus_memory_info
from utils import BinaryIndicatorsMetric, AverageMeter
from utils import BCEDiceLoss, SoftDiceLoss, DiceLoss
from utils import LRScheduler

# ========================================================================================
# TRANSFORMER CONFIGURATION SEARCH SPACE
# ========================================================================================

# Candidate values for transformer hyperparameters
TRANSFORMER_D_MODEL_CHOICES = [64, 128, 256, 512]  # Hidden dimension
TRANSFORMER_N_HEAD_CHOICES = [2, 4, 8]  # Number of attention heads
TRANSFORMER_EXPANSION_CHOICES = [2, 4, 8]  # FFN expansion ratio (d_ff = d_model * expansion)

def get_transformer_config_choices():
    """Get all valid transformer configuration choices"""
    configs = []
    for d_model in TRANSFORMER_D_MODEL_CHOICES:
        for n_head in TRANSFORMER_N_HEAD_CHOICES:
            for expansion in TRANSFORMER_EXPANSION_CHOICES:
                # Constraint: d_model must be divisible by n_head
                if d_model % n_head == 0:
                    configs.append({
                        'd_model': d_model,
                        'n_head': n_head,
                        'expansion': expansion,
                        'd_ff': d_model * expansion
                    })
    return configs

TRANSFORMER_CONFIG_CHOICES = get_transformer_config_choices()
NUM_TRANSFORMER_CONFIGS = len(TRANSFORMER_CONFIG_CHOICES)

print(f"Total transformer configurations: {NUM_TRANSFORMER_CONFIGS}")
for i, config in enumerate(TRANSFORMER_CONFIG_CHOICES[:5]):  # Print first 5
    print(f"  Config {i}: d_model={config['d_model']}, n_head={config['n_head']}, "
          f"expansion={config['expansion']}, d_ff={config['d_ff']}")


# ========================================================================================
# COMPLEXITY CALCULATION FOR TRANSFORMERS
# ========================================================================================

def calculate_transformer_complexity(config, seq_len=256, num_layers=6):
    """
    Calculate FLOPs and parameters for a transformer block configuration
    
    Args:
        config: Dict with d_model, n_head, expansion, d_ff
        seq_len: Sequence length (H*W for images)
        num_layers: Number of transformer layers
    
    Returns:
        tuple: (flops, params) in millions
    """
    d_model = config['d_model']
    n_head = config['n_head']
    d_ff = config['d_ff']
    
    # Multi-head attention FLOPs per layer
    # Q, K, V projections: 3 * seq_len * d_model * d_model
    qkv_flops = 3 * seq_len * d_model * d_model
    
    # Attention score calculation (Q @ K^T): seq_len * seq_len * d_model
    # Applying attention to V (scores @ V): seq_len * seq_len * d_model
    attn_flops = 2 * seq_len * seq_len * d_model
    
    # Output linear projection: seq_len * d_model * d_model
    output_proj_flops = seq_len * d_model * d_model
    
    mhsa_flops = qkv_flops + attn_flops + output_proj_flops

    # 2. Feed-Forward Network (FFN)
    # First linear layer: seq_len * d_model * d_ff
    # Second linear layer: seq_len * d_ff * d_model
    ffn_flops = seq_len * d_model * d_ff + seq_len * d_ff * d_model
    
    # Total FLOPs per layer
    flops_per_layer = mhsa_flops + ffn_flops
    
    # Total FLOPs for all layers, converted to millions
    total_flops = flops_per_layer * num_layers / 1e6
    
    # --- Parameter Calculation (per layer) ---
    
    # 1. MHSA Parameters
    # Q, K, V weights and biases: 3 * (d_model * d_model)
    qkv_params = 3 * d_model * d_model
    
    # Output projection weights and biases
    output_proj_params = d_model * d_model
    
    # 2. FFN Parameters
    # Two linear layers: (d_model * d_ff) + (d_ff * d_model)
    ffn_params = d_model * d_ff + d_ff * d_model
    
    # 3. Layer Normalization Parameters
    # Standard transformer block has 2 LayerNorms (one before MHSA, one before FFN)
    # Each LayerNorm has 2 learnable params (gamma, beta) per feature
    ln_params = 2 * (2 * d_model)
    
    # Total parameters per layer
    params_per_layer = qkv_params + output_proj_params + ffn_params + ln_params
    
    # Total params for all layers, converted to millions
    total_params = params_per_layer * num_layers / 1e6
    
    return total_flops, total_params


# Pre-calculate complexity for all configurations
TRANSFORMER_COMPLEXITY_LOOKUP = {}
for i, config in enumerate(TRANSFORMER_CONFIG_CHOICES):
    flops, params = calculate_transformer_complexity(config)
    TRANSFORMER_COMPLEXITY_LOOKUP[i] = {
        'flops': flops,
        'params': params,
        'complexity_score': flops + 0.1 * params  # Weighted combination
    }


# GUMBEL-SOFTMAX FOR DISCRETE SAMPLING

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax sampling to create discrete-like but differentiable sampling.
    Helps avoid convergence to same probability values.
    
    Args:
        logits: [..., num_classes] unnormalized log probabilities
        temperature: Temperature parameter (lower = more discrete)
        hard: If True, return one-hot, but backprop through soft sample
    
    Returns:
        Sampled probabilities
    """
    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    
    # Add noise and apply softmax with temperature
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if hard:
        # Straight-through estimator: forward pass is one-hot, backward pass is soft
        y_hard = torch.zeros_like(y_soft)
        y_hard.scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        y = y_hard - y_soft.detach() + y_soft  # Gradient flows through y_soft
    else:
        y = y_soft
    
    return y


# MULTI-OBJECTIVE PARETO OPTIMIZATION WITH BITSTRING ENCODING

def encode_architecture_to_bitstring(model):
    """
    Encode current architecture to bitstring representation.
    x ∈ {0,1}^n where n = total number of architecture decisions
    
    Bitstring structure:
    - Transformer connections: [conn_0, conn_1, ..., conn_k] (0=OFF, 1=ON)
    - Transformer configs: [config_bits for each connection]
    
    Returns:
        bitstring: Binary tensor representing architecture
        decode_info: Dict with information for decoding
    """
    bitstring = []
    decode_info = {
        'connection_indices': [],
        'config_indices': []
    }
    
    # Encode transformer connections (binary: ON/OFF)
    if hasattr(model, 'alphas_transformer_connections'):
        conn_probs = F.softmax(model.alphas_transformer_connections, dim=-1)
        conn_decisions = (conn_probs[:, 1] > 0.5).float()  # 1 if ON, 0 if OFF
        bitstring.append(conn_decisions)
        decode_info['connection_indices'] = list(range(len(conn_decisions)))
    
    # Encode transformer configurations (multi-bit encoding)
    if hasattr(model, 'alphas_transformer_configs'):
        config_probs = F.softmax(model.alphas_transformer_configs, dim=-1)
        # For each connection, encode selected config as binary
        num_connections = config_probs.shape[0]
        bits_per_config = int(np.ceil(np.log2(NUM_TRANSFORMER_CONFIGS)))
        
        for conn_idx in range(num_connections):
            selected_config = torch.argmax(config_probs[conn_idx]).item()
            # Convert to binary representation
            config_bits = [int(b) for b in format(selected_config, f'0{bits_per_config}b')]
            bitstring.extend(config_bits)
            decode_info['config_indices'].append((conn_idx, bits_per_config))
    
    # Convert to tensor
    if bitstring:
        if isinstance(bitstring[0], torch.Tensor):
            bitstring_tensor = torch.cat([b if b.dim() > 0 else b.unsqueeze(0) for b in bitstring])
        else:
            bitstring_tensor = torch.tensor(bitstring, dtype=torch.float32, device=model.alphas_transformer_connections.device)
    else:
        bitstring_tensor = torch.tensor([], dtype=torch.float32)
    
    return bitstring_tensor, decode_info


def compute_pareto_objectives(model, dice_loss, args):
    """
    Compute 3 objectives for Pareto optimization:
    f(x) = [f1(x), f2(x), f3(x)]
    
    All objectives are to be MINIMIZED:
    - f1(x): Segmentation error (1 - Dice)
    - f2(x): Model complexity (FLOPs + params)
    - f3(x): Architecture diversity penalty
    
    Args:
        model: NAS model
        dice_loss: Dice loss value
        args: Arguments
    
    Returns:
        objectives: Dict with f1, f2, f3 values
    """
    # Objective 1: Segmentation accuracy (minimize error)
    f1 = dice_loss  # Already in [0, 1], lower is better
    
    # Objective 2: Complexity (minimize FLOPs + params + number of connections)
    # Per-connection + Gumbel-Softmax CÁCH 1: Tách complexity theo từng connection - f2 = sum_i (p_i * cost_i)
    if hasattr(model, 'alphas_transformer_connections'):
        # Option to use Gumbel-Softmax for discrete sampling
        use_gumbel = getattr(args, 'use_gumbel_softmax', False)
        temperature = getattr(args, 'gumbel_temperature', 1.0)
        
        if use_gumbel:
            conn_probs = gumbel_softmax_sample(
                model.alphas_transformer_connections, 
                temperature=temperature, 
                hard=False
            )
        else:
            conn_probs = F.softmax(model.alphas_transformer_connections, dim=-1)
        
        probs_on = conn_probs[:, 1]
        
        #  CÁCH 3: Inject positional bias vào cost để phá symmetry
        # connection_cost_i = base_cost * (1 + γ * depth_i)
        positional_bias_factor = getattr(args, 'positional_bias_factor', 0.0)  # γ
        
        # Tính complexity cho từng connection riêng biệt (QUAN TRỌNG cho differentiation)
        # f2 = sum_i (p_i * cost_i) thay vì f2 = sum(p_i) * total_cost
        per_connection_complexity = []
        
        if hasattr(model, 'alphas_transformer_configs'):
            config_probs = F.softmax(model.alphas_transformer_configs, dim=-1)
            
            for conn_idx in range(config_probs.shape[0]):
                # Expected complexity for this specific connection
                expected_complexity = torch.tensor(0.0, device=args.device)
                for config_idx, prob in enumerate(config_probs[conn_idx]):
                    base_complexity = TRANSFORMER_COMPLEXITY_LOOKUP[config_idx]['complexity_score']
                    
                    #  CÁCH 3: Apply positional bias - cost_i = base_cost * (1 + γ * i)
                    # Phá vỡ symmetry: cùng config nhưng khác vị trí có cost khác nhau
                    positional_multiplier = 1.0 + positional_bias_factor * conn_idx
                    biased_complexity = base_complexity * positional_multiplier
                    
                    expected_complexity += prob * biased_complexity
                
                #  CÁCH 1 (QUAN TRỌNG): f2_i = p_i * cost_i cho từng connection
                # Gradient sẽ khác nhau nếu cost_i khác nhau
                weighted_complexity = probs_on[conn_idx] * expected_complexity
                per_connection_complexity.append(weighted_complexity)
        else:
            # If no config search, use different base complexity per position
            for conn_idx in range(len(probs_on)):
                base_complexity = 50.0  # Base complexity
                
                #  CÁCH 3: Apply positional bias
                positional_multiplier = 1.0 + positional_bias_factor * conn_idx
                connection_complexity = base_complexity * positional_multiplier
                
                #  CÁCH 1: f2_i = p_i * cost_i
                weighted_complexity = probs_on[conn_idx] * connection_complexity
                per_connection_complexity.append(weighted_complexity)
        
        # Sum all per-connection complexities
        total_complexity = sum(per_connection_complexity)
        
        # Expected number of active connections
        expected_num_connections = torch.sum(probs_on)
        
        # Normalize
        num_connections = probs_on.shape[0]
        connection_ratio = expected_num_connections / num_connections  # [0, 1]
        complexity_ratio = torch.clamp(total_complexity / (100.0 * num_connections), 0, 1)
        
        # Weighted combination: 50% connection count, 50% computational cost
        f2 = 0.5 * connection_ratio + 0.5 * complexity_ratio
    else:
        f2 = torch.tensor(0.0, device=args.device)
    
    # Objective 3: Connection differentiation penalty (minimize indecision)
    # Enforce that the p_i are not all the same
    if hasattr(model, 'alphas_transformer_connections'):
        conn_probs = F.softmax(model.alphas_transformer_connections, dim=-1)
        probs_on = conn_probs[:, 1]
        
        # Choose diversity loss type
        diversity_loss_type = getattr(args, 'diversity_loss_type', 'indecision')  # 'indecision', 'variance', 'repulsion'
        
        if diversity_loss_type == 'variance':
            #  CÁCH 2a: Variance loss - L_div = -Var(p)
            # Maximize variance = minimize negative variance
            # Variance cao = các p_i khác nhau nhiều (TỐT)
            # Variance thấp = các p_i gần giống nhau (XẤU)
            mean_prob = torch.mean(probs_on)
            variance = torch.mean((probs_on - mean_prob) ** 2)
            
            # Normalize variance to [0, 0.25] (max variance when half 0, half 1)
            # Then convert to penalty: minimize (-variance)
            # High variance = low penalty (GOOD)
            # Low variance = high penalty (BAD)
            max_variance = 0.25  # Maximum possible variance for binary [0,1]
            normalized_variance = variance / max_variance  # [0, 1]
            
            f3 = 1.0 - normalized_variance  # Minimize = encourage high variance
            
        elif diversity_loss_type == 'repulsion':
            #  CÁCH 2b: Repulsion loss - L_div = sum_{i≠j} exp(-|p_i - p_j|)
            # Penalize probabilities that are too close to each other
            # Force different connections to have different probabilities
            repulsion_loss = torch.tensor(0.0, device=args.device)
            num_connections = len(probs_on)
            
            for i in range(num_connections):
                for j in range(i + 1, num_connections):
                    # Penalize small distances between probabilities
                    distance = torch.abs(probs_on[i] - probs_on[j])
                    repulsion_loss += torch.exp(-distance * 5.0)  # Scale factor 5.0
            
            # Normalize by number of pairs
            num_pairs = num_connections * (num_connections - 1) / 2
            if num_pairs > 0:
                repulsion_loss = repulsion_loss / num_pairs
            
            # repulsion_loss is in [0, 1], minimize it
            f3 = repulsion_loss
            
        else:  # 'indecision' (original)
            # Original: Penalty for being close to 0.5 (indecisive)
            distances_from_middle = torch.abs(probs_on - 0.5)  # [0, 0.5]
            indecision_penalty = 0.5 - distances_from_middle  # [0, 0.5]
            f3 = torch.mean(indecision_penalty) * 2.0  # Scale to [0, 1]
    else:
        f3 = torch.tensor(0.0, device=args.device)
    
    return {'f1': f1, 'f2': f2, 'f3': f3}


def compute_pareto_loss_scalarization(objectives, args, method='weighted_sum'):
    """
    Scalarize multi-objective problem using various methods.
    
    Methods:
    - 'weighted_sum': λ1*f1 + λ2*f2 + λ3*f3
    - 'tchebycheff': Minimize max weighted deviation from ideal point
    - 'augmented_tchebycheff': Tchebycheff + penalty term
    
    Args:
        objectives: Dict with f1, f2, f3
        args: Arguments with weights
        method: Scalarization method
    
    Returns:
        scalar_loss: Single scalar loss value
    """
    f1, f2, f3 = objectives['f1'], objectives['f2'], objectives['f3']
    
    # Weights (sum to 1 for proper scalarization)
    w1 = getattr(args, 'pareto_weight_dice', 0.4)
    w2 = getattr(args, 'pareto_weight_complexity', 0.4)
    w3 = getattr(args, 'pareto_weight_connection', 0.2)
    
    # Normalize weights
    total_w = w1 + w2 + w3
    w1, w2, w3 = w1/total_w, w2/total_w, w3/total_w
    
    if method == 'weighted_sum':
        # Linear scalarization
        scalar_loss = w1 * f1 + w2 * f2 + w3 * f3
        
    elif method == 'tchebycheff':
        # Tchebycheff scalarization (better for non-convex Pareto fronts)
        # min max_i { λi * |fi - zi*| }
        # where zi* is the ideal point (here assumed to be 0)
        weighted_objectives = torch.stack([w1 * f1, w2 * f2, w3 * f3])
        scalar_loss = torch.max(weighted_objectives)
        
    elif method == 'augmented_tchebycheff':
        # Augmented Tchebycheff
        weighted_objectives = torch.stack([w1 * f1, w2 * f2, w3 * f3])
        rho = 0.05  # Augmentation parameter
        scalar_loss = torch.max(weighted_objectives) + rho * (f1 + f2 + f3)
    
    else:
        raise ValueError(f"Unknown scalarization method: {method}")
    
    return scalar_loss


def decode_bitstring_to_architecture(bitstring, decode_info, model):
    """
    Decode bitstring back to architecture parameters.
    Updates model's alpha parameters based on bitstring.
    
    Args:
        bitstring: Binary tensor
        decode_info: Decoding information from encoding
        model: Model to update
    """
    idx = 0
    
    # Decode transformer connections
    if 'connection_indices' in decode_info and len(decode_info['connection_indices']) > 0:
        num_connections = len(decode_info['connection_indices'])
        conn_bits = bitstring[idx:idx+num_connections]
        
        with torch.no_grad():
            for i, bit in enumerate(conn_bits):
                if bit > 0.5:  # ON
                    model.alphas_transformer_connections[i, 0] = -1.0  # OFF logit
                    model.alphas_transformer_connections[i, 1] = 1.0   # ON logit
                else:  # OFF
                    model.alphas_transformer_connections[i, 0] = 1.0   # OFF logit
                    model.alphas_transformer_connections[i, 1] = -1.0  # ON logit
        
        idx += num_connections
    
    # Decode transformer configurations
    if 'config_indices' in decode_info:
        for conn_idx, bits_per_config in decode_info['config_indices']:
            config_bits = bitstring[idx:idx+bits_per_config]
            
            # Convert binary to config index
            config_idx = 0
            for i, bit in enumerate(config_bits):
                if bit > 0.5:
                    config_idx += 2 ** (bits_per_config - 1 - i)
            
            # Update alpha to favor this configuration
            with torch.no_grad():
                model.alphas_transformer_configs[conn_idx, :] = -1.0
                if config_idx < NUM_TRANSFORMER_CONFIGS:
                    model.alphas_transformer_configs[conn_idx, config_idx] = 2.0
            
            idx += bits_per_config


# MAIN TRAINING FUNCTION

def main(args):
    ############    init config ################
    
    #################### init logger ###################################
    log_dir = 'search_exp/{}/{}/{}_{}'.format(args.model, args.dataset, args.note, time.strftime("%Y%m%d-%H%M%S"))

    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-Search-Pareto-V2'.format(args.model))
    args.save_path = log_dir
    args.save_tbx_log = args.save_path + '/tbx_log'
    writer = SummaryWriter(args.save_tbx_log)
    
    # Initialize Weights & Biases
    wandb.init(
        project="hct-net-ablation",
        name=f"{args.model}_{args.dataset}_{args.note}_{time.strftime('%Y%m%d-%H%M%S')}",
        config=vars(args),
        dir=log_dir,
        tags=["pareto_optimization", "transformer_nas", "3_objectives"]
    )
    
    ##################### init device #################################
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    args.use_cuda = args.gpus > 0 and torch.cuda.is_available()
    args.multi_gpu = args.gpus > 1 and torch.cuda.is_available()
    args.device = torch.device('cuda:0' if args.use_cuda else 'cpu')
    if args.use_cuda:
        torch.cuda.manual_seed(args.manualSeed)
        cudnn.enabled = True
        cudnn.benchmark = True
    setting = {k: v for k, v in args._get_kwargs()}
    logger.info(setting)

    ####################### init dataset ###########################################
    logger.info("Dataset for search is {}".format(args.dataset))
    train_dataset = datasets_dict[args.dataset](args, args.dataset_root, split='train')
    val_dataset = datasets_dict[args.dataset](args, args.dataset_root, split='valid')

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    
    # init loss
    if args.loss == 'bce':
        criterion = nn.BCELoss()
    elif args.loss == 'bcelog':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "softdice":
        criterion = SoftDiceLoss()
    elif args.loss == 'bcedice':
        criterion = BCEDiceLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    if args.use_cuda:
        logger.info("load criterion to gpu !")
    criterion = criterion.to(args.device)
    
    ######################## init model ############################################
    switches_normal = []
    switches_down = []
    switches_up = []
    switches_transformer = []
    nums_mixop = sum([2 + i for i in range(args.meta_node_num)])
    for i in range(nums_mixop):
        switches_normal.append([True for j in range(len(CellPos))])
    for i in range(nums_mixop):
        switches_down.append([True for j in range(len(CellLinkDownPos))])
    for i in range(nums_mixop):
        switches_up.append([True for j in range(len(CellLinkUpPos))])
    
    # Initialize transformer switches for configurations
    for i in range(args.layers):
        switches_transformer.append([True for j in range(NUM_TRANSFORMER_CONFIGS)])

    original_train_batch = args.train_batch
    original_val_batch = args.val_batch

    #############################select model################################
    args.model = "UnetLayer{}".format(args.layers)
    sp_train_batch = original_train_batch
    sp_val_batch = original_val_batch
    sp_lr = args.lr
    sp_epoch = args.epochs
    early_fix_arch = args.early_fix_arch
    gen_max_child_flag = args.gen_max_child_flag
    random_sample = args.random_sample
    
    logger.info(f"Building model with {args.layers} layers")
    logger.info(f"Expected transformer connections: {(args.layers - 1) // 2}")
    logger.info(f"Transformer configuration search space: {NUM_TRANSFORMER_CONFIGS} options")

    ###################################dataset#########################################
    train_queue = data.DataLoader(train_dataset,
                                  batch_size=sp_train_batch,
                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                  pin_memory=True,
                                  num_workers=args.num_workers
                                  )
    val_queue = data.DataLoader(train_dataset,
                                batch_size=sp_train_batch,
                                sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
                                pin_memory=True,
                                num_workers=args.num_workers
                                )

    logger.info(
        "model:{} epoch:{} lr:{} train_batch:{} val_batch:{}".format(args.model, sp_epoch, sp_lr, sp_train_batch,
                                                                     sp_val_batch))

    model = get_models(args, switches_normal, switches_down, switches_up, switches_transformer, early_fix_arch, 
                       gen_max_child_flag, random_sample)

    # === INITIALIZE TRANSFORMER CONFIGURATION ALPHAS ===
    if hasattr(model, 'alphas_transformer_configs'):
        num_connections = (args.layers - 1) // 2
        # Initialize with small random values
        with torch.no_grad():
            # Slight bias toward mid-range configurations
            for i in range(num_connections):
                model.alphas_transformer_configs[i, :] = torch.randn(NUM_TRANSFORMER_CONFIGS) * 0.01
        
        logger.info("=" * 80)
        logger.info("TRANSFORMER CONFIGURATION NAS INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"Initialized {num_connections} transformer configuration alphas")
        logger.info(f"Search space: {NUM_TRANSFORMER_CONFIGS} configurations")
        logger.info(f"Parameters: d_model={TRANSFORMER_D_MODEL_CHOICES}, "
                   f"n_head={TRANSFORMER_N_HEAD_CHOICES}, "
                   f"expansion={TRANSFORMER_EXPANSION_CHOICES}")
        logger.info("=" * 80 + "\n")
    
    # === INITIALIZE TRANSFORMER CONNECTION ALPHAS ===
    if hasattr(model, 'alphas_transformer_connections') and hasattr(args, 'transformer_init_bias'):
        with torch.no_grad():
            bias = args.transformer_init_bias
            
            # Add positional-dependent initialization
            # alpha_i += ε * i để các connection khác nhau có initialization khác nhau
            positional_init_scale = getattr(args, 'positional_init_scale', 0.1)  # ε
            
            for i in range(len(model.alphas_transformer_connections)):
                # Base bias
                base_off_logit = -bias
                base_on_logit = bias
                
                # Add positional bias - break symmetry
                # Different connections at different positions have different initializations
                positional_offset = positional_init_scale * i
                
                model.alphas_transformer_connections[i, 0] = base_off_logit - positional_offset  # OFF
                model.alphas_transformer_connections[i, 1] = base_on_logit + positional_offset   # ON
            
            init_probs = F.softmax(model.alphas_transformer_connections, dim=-1)
            logger.info("=" * 80)
            logger.info("TRANSFORMER CONNECTION INITIALIZATION WITH POSITIONAL BIAS")
            logger.info("=" * 80)
            logger.info(f"Initialized {len(model.alphas_transformer_connections)} transformer connections")
            logger.info(f"Base bias value: {bias}")
            logger.info(f"Positional scale (ε): {positional_init_scale}")
            logger.info(f"Initial ON probabilities: {[f'{p:.3f}' for p in init_probs[:, 1].tolist()]}")
            logger.info(f"Mean ON probability: {init_probs[:, 1].mean():.3f}")
            logger.info(f"Std ON probability: {init_probs[:, 1].std():.3f} (higher = more diversity)")
            logger.info("=" * 80 + "\n")
    
    # Initialize gradients
    for v in model.parameters():
        if v.requires_grad:
            if v.grad is None:
                v.grad = torch.zeros_like(v)
    
    model.alphas_up.grad = torch.zeros_like(model.alphas_up)
    model.alphas_down.grad = torch.zeros_like(model.alphas_down)
    model.alphas_normal.grad = torch.zeros_like(model.alphas_normal)
    model.alphas_network.grad = torch.zeros_like(model.alphas_network)
    
    if hasattr(model, 'alphas_transformer_connections'):
        model.alphas_transformer_connections.grad = torch.zeros_like(model.alphas_transformer_connections)
    
    if hasattr(model, 'alphas_transformer_configs'):
        model.alphas_transformer_configs.grad = torch.zeros_like(model.alphas_transformer_configs)

    # Setup optimizer parameters
    wo_wd_params = []
    wo_wd_param_names = []
    network_params = []
    network_param_names = []
    
    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            for key, value in mod.named_parameters():
                wo_wd_param_names.append(name + '.' + key)

    for key, value in model.named_parameters():
        if "alphas" not in key:
            if value.requires_grad:
                if key in wo_wd_param_names:
                    wo_wd_params.append(value)
                else:
                    network_params.append(value)
                    network_param_names.append(key)

    weight_parameters = [
        {'params': network_params,
         'lr': args.lr,
         'weight_decay': args.weight_decay },
        {'params': wo_wd_params,
         'lr': args.lr,
         'weight_decay': 0.},
    ]

    save_model_path = os.path.join(args.save_path, 'single')
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if args.multi_gpu:
        logger.info('use: %d gpus', args.gpus)
        model = nn.DataParallel(model)
    model = model.to(args.device)
    logger.info('param size = %fMB', calc_parameters_count(model))

    # init optimizer for arch parameters and weight parameters
    optimizer_arch = torch.optim.Adam(model.arch_parameters(), lr=args.arch_lr, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    optimizer_weight = torch.optim.SGD(weight_parameters, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_weight, sp_epoch, eta_min=args.lr_min)

    #################################### train and val ########################
    start_epoch = 0

    if args.resume is not None and args.resume != 'None':
        if os.path.isfile(args.resume):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=args.device)
            start_epoch = checkpoint['epoch']
            optimizer_arch.load_state_dict(checkpoint['optimizer_arch'])
            optimizer_weight.load_state_dict(checkpoint['optimizer_weight'])
            model_inner = model.module if hasattr(model, 'module') else model
            model_inner.load_alphas(checkpoint['alphas_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

    logger.info("="*80)
    logger.info("LOSS CONFIGURATION: PARETO V2 WITH PROBABILITY DIFFERENTIATION")
    logger.info("="*80)
    logger.info("✓ Multi-objective optimization with 3 objectives (all minimized):")
    logger.info("  - f1(x): Segmentation error (Dice loss)")
    logger.info("  - f2(x): Model complexity (FLOPs + params)")
    logger.info("  - f3(x): Probability differentiation")
    logger.info(f"✓ Scalarization method: {getattr(args, 'pareto_scalarization', 'weighted_sum')}")
    logger.info(f"✓ Weights: λ1={getattr(args, 'pareto_weight_dice', 0.3)}, "
               f"λ2={getattr(args, 'pareto_weight_complexity', 0.3)}, "
               f"λ3={getattr(args, 'pareto_weight_connection', 0.4)}")
    logger.info("="*80)
    logger.info(" PROBABILITY DIFFERENTIATION TECHNIQUES:")
    logger.info("="*80)
    logger.info(" Cách 1: Per-Connection Complexity (f2 = Σ p_i * cost_i)")
    logger.info(f"    → Positional bias factor γ = {getattr(args, 'positional_bias_factor', 0.0)}")
    logger.info(f"    → Each connection has unique gradient based on its cost")
    logger.info("")
    logger.info(f" Cách 2: Diversity Loss Type = {getattr(args, 'diversity_loss_type', 'indecision')}")
    if args.diversity_loss_type == 'variance':
        logger.info("    → Variance loss: L_div = -Var(p) → encourages different probabilities")
    elif args.diversity_loss_type == 'repulsion':
        logger.info("    → Repulsion loss: L_div = Σ exp(-|p_i - p_j|) → forces separation")
    else:
        logger.info("    → Indecision penalty: encourages decisive ON/OFF")
    logger.info("")
    logger.info(f" Cách 3: Positional Bias (breaks symmetry)")
    logger.info(f"    → Alpha initialization scale ε = {getattr(args, 'positional_init_scale', 0.1)}")
    logger.info(f"    → Cost bias factor γ = {getattr(args, 'positional_bias_factor', 0.0)}")
    logger.info(f"    → alpha_i += ε*i, cost_i = base_cost*(1+γ*i)")
    logger.info("")
    logger.info(f" Cách 4: Gumbel-Softmax Sampling = {args.use_gumbel_softmax}")
    if args.use_gumbel_softmax:
        logger.info(f"    → Temperature = {args.gumbel_temperature}")
        logger.info(f"    → Annealing = {args.gumbel_anneal}")
        if args.gumbel_anneal:
            logger.info(f"    → Min temperature = {args.gumbel_temp_min}")
        logger.info("    → Discrete-like sampling prevents uniform convergence")
    logger.info("="*80 + "\n")

    max_value = 0
    
    for epoch in range(start_epoch, sp_epoch):
        logger.info('################Epoch: %d lr %e######################', epoch, scheduler.get_last_lr()[0])
        
        #  CÁCH 4: Gumbel-Softmax temperature annealing
        if args.use_gumbel_softmax and args.gumbel_anneal:
            # Linearly anneal temperature from initial to minimum
            progress = epoch / sp_epoch
            args.gumbel_temperature = max(
                args.gumbel_temp_min,
                args.gumbel_temperature * (1.0 - progress) + args.gumbel_temp_min * progress
            )
            logger.info(f"Gumbel temperature: {args.gumbel_temperature:.3f}")

        if args.early_fix_arch:
            # Fix architecture for cell operations (same as before)
            _fix_cell_operations(model, args, logger, epoch)
        
        # Train for one epoch
        if epoch < args.arch_after:
            train_stats = train(args, train_queue, val_queue, model, criterion, 
                              optimizer_weight, optimizer_arch,
                              train_arch=False, epoch=epoch, total_epochs=sp_epoch)
        else:
            train_stats = train(args, train_queue, val_queue, model, criterion, 
                              optimizer_weight, optimizer_arch,
                              train_arch=True, epoch=epoch, total_epochs=sp_epoch)

        weight_loss_avg, arch_loss_avg, complexity_loss_avg, connection_loss_avg, \
            mr, ms, mp, mf, mjc, md, macc, epoch_time = train_stats

        logger.info("Epoch:{} WeightLoss:{:.3f}  ArchLoss:{:.3f}".format(epoch, weight_loss_avg, arch_loss_avg))
        logger.info("         Acc:{:.3f}   Dice:{:.3f}  Jc:{:.3f}".format(macc, md, mjc))
        logger.info("         f2(Complexity):{:.4f}  f3(Diversity):{:.4f}".format(
            complexity_loss_avg, connection_loss_avg))
        
        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train/weight_loss': weight_loss_avg,
            'train/arch_loss': arch_loss_avg,
            'train/complexity_loss': complexity_loss_avg,
            'train/connection_loss': connection_loss_avg,
            'train/accuracy': macc,
            'train/dice': md,
            'train/jaccard': mjc,
            'lr': scheduler.get_last_lr()[0],
        })
        
        # Log architecture decisions
        if epoch >= args.arch_after and epoch % 5 == 0:
            _log_architecture_decisions(model, logger, writer, epoch)
        
        # Step scheduler
        scheduler.step()

        # Write to tensorboard
        writer.add_scalar('Train/W_loss', weight_loss_avg, epoch)
        writer.add_scalar('Train/A_loss', arch_loss_avg, epoch)
        writer.add_scalar('Train/Dice', md, epoch)
        writer.add_scalar('Train/Complexity_loss', complexity_loss_avg, epoch)
        writer.add_scalar('Train/Connection_loss', connection_loss_avg, epoch)

        # Validation
        if (epoch + 1) % args.infer_epoch == 0:
            genotype = model.genotype()
            logger.info('genotype = %s', genotype)
            val_loss, (vmr, vms, vmp, vmf, vmjc, vmd, vmacc) = infer(args, model, val_queue, criterion)
            logger.info("ValLoss:{:.3f} ValAcc:{:.3f}  ValDice:{:.3f} ValJc:{:.3f}".format(
                val_loss, vmacc, vmd, vmjc))
            
            writer.add_scalar('Val/loss', val_loss, epoch)
            writer.add_scalar('Val/dice', vmd, epoch)
            writer.add_scalar('Val/jaccard', vmjc, epoch)
            
            wandb.log({
                'val/loss': val_loss,
                'val/accuracy': vmacc,
                'val/dice': vmd,
                'val/jaccard': vmjc,
            })

            is_best = vmjc >= max_value
            max_value = max(max_value, vmjc)
            state = {
                'epoch': epoch,
                'max_value': max_value,
                'optimizer_arch': optimizer_arch.state_dict(),
                'optimizer_weight': optimizer_weight.state_dict(),
                'scheduler': scheduler.state_dict(),
                'state_dict': model.state_dict(),
                'alphas_dict': model.alphas_dict(),
            }

            logger.info("epoch:{} best:{} max_value:{}".format(epoch, is_best, max_value))
            wandb.log({'val/max_jaccard': max_value, 'val/is_best': 1 if is_best else 0})
            
            # Checkpoint saving disabled
            # torch.save(state, os.path.join(save_model_path, "checkpoint.pth.tar"))
            # if is_best:
            #     torch.save(state, os.path.join(save_model_path, "model_best.pth.tar"))

    # Final architecture report
    _print_final_architecture(model, logger)
    
    logger.info('param size = %fMB', calc_parameters_count(model))

    writer.close()


def train(args, train_queue, val_queue, model, criterion, optimizer_weight, optimizer_arch, 
          train_arch, epoch=0, total_epochs=20):
    """
    Training function with Pareto V2 optimization (3 objectives)
    """
    Train_recoder = BinaryIndicatorsMetric()
    w_loss_recoder = AverageMeter()
    a_loss_recoder = AverageMeter()
    complexity_loss_recoder = AverageMeter()
    connection_loss_recoder = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    end = time.time()

    for step, (input, target, _) in enumerate(train_queue):
        data_time.update(time.time() - end)

        input = input.to(args.device)
        target = target.to(args.device)

        # Weight optimization
        optimizer_weight.zero_grad()
        preds = model(input, target, criterion)
        assert isinstance(preds, list)
        preds = [pred.view(pred.size(0), -1) for pred in preds]
        target_flat = target.view(target.size(0), -1)

        torch.cuda.empty_cache()

        if args.deepsupervision:
            w_loss = sum(criterion(pred, target_flat) for pred in preds)
        else:
            w_loss = criterion(preds[-1], target_flat)
        
        w_loss.backward()

        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)

        optimizer_weight.step()
        w_loss_recoder.update(w_loss.item(), 1)
        Train_recoder.update(labels=target_flat, preds=preds[-1], n=1)

        # Architecture optimization
        if train_arch:
            try:
                input_search, target_search, _ = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(val_queue)
                input_search, target_search, _ = next(valid_queue_iter)
            
            input_search = input_search.to(args.device)
            target_search = target_search.to(args.device)

            optimizer_arch.zero_grad()
            archs_preds = model(input_search, target_search, criterion)
            archs_preds = [pred.view(pred.size(0), -1) for pred in archs_preds]
            target_search_flat = target_search.view(target_search.size(0), -1)
            
            torch.cuda.empty_cache()
            
            # Compute dice loss
            if args.deepsupervision:
                dice_loss = sum(criterion(pred, target_search_flat) for pred in archs_preds)
            else:
                dice_loss = criterion(archs_preds[-1], target_search_flat)
            
            # Encode architecture to bitstring
            bitstring, decode_info = encode_architecture_to_bitstring(model)
            
            # Compute 3 Pareto objectives
            objectives = compute_pareto_objectives(model, dice_loss, args)
            
            # Scalarize multi-objective problem
            scalarization_method = getattr(args, 'pareto_scalarization', 'weighted_sum')
            a_loss = compute_pareto_loss_scalarization(objectives, args, method=scalarization_method)
            
            # Record losses
            a_loss_recoder.update(a_loss.item(), 1)
            complexity_loss_recoder.update(objectives['f2'].item(), 1)
            connection_loss_recoder.update(objectives['f3'].item(), 1)

            # Backward pass for architecture parameters
            a_loss.backward()
            
            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)

            optimizer_arch.step()

        batch_time.update(time.time() - end)
        end = time.time()

    weight_loss_avg = w_loss_recoder.avg
    arch_loss_avg = a_loss_recoder.avg if train_arch else 0
    complexity_loss_avg = complexity_loss_recoder.avg if complexity_loss_recoder.count > 0 else 0
    connection_loss_avg = connection_loss_recoder.avg if connection_loss_recoder.count > 0 else 0
    
    mr, ms, mp, mf, mjc, md, macc = Train_recoder.get_avg

    return (weight_loss_avg, arch_loss_avg, complexity_loss_avg, connection_loss_avg,
            mr, ms, mp, mf, mjc, md, macc, batch_time)


def infer(args, model, val_queue, criterion):
    batch_time = AverageMeter()
    OtherVal = BinaryIndicatorsMetric()
    val_loss = AverageMeter()
    model.eval()

    with torch.no_grad():
        end = time.time()
        for step, (input, target, _) in tqdm(enumerate(val_queue)):
            input = input.to(args.device)
            target = target.to(args.device)
            preds = model(input, target, criterion)
            preds = [pred.view(pred.size(0), -1) for pred in preds]
            target_flat = target.view(target.size(0), -1)
            
            if args.deepsupervision:
                loss = sum(criterion(pred, target_flat) for pred in preds)
            else:
                loss = criterion(preds[-1], target_flat)
            
            val_loss.update(loss.item(), 1)
            OtherVal.update(labels=target_flat, preds=preds[-1], n=1)

        batch_time.update(time.time() - end)
        
    return val_loss.avg, OtherVal.get_avg


def _fix_cell_operations(model, args, logger, epoch):
    """Fix cell operations when confident enough"""
    # Normal cells
    if len(model.fix_arch_normal_index.keys()) > 0:
        for key, value_lst in model.fix_arch_normal_index.items():
            model.alphas_normal.data[key, :] = value_lst[1]

    sort_log_alpha_normal = torch.topk(F.softmax(model.alphas_normal.data, dim=-1), 2)
    argmax_index_normal = (sort_log_alpha_normal[0][:, 0] - sort_log_alpha_normal[0][:, 1] >= 0.3)

    for id in range(argmax_index_normal.size(0)):
        if argmax_index_normal[id] == 1 and id not in model.fix_arch_normal_index.keys():
            model.fix_arch_normal_index[id] = [sort_log_alpha_normal[1][id, 0].item(),
                                               model.alphas_normal.detach().clone()[id, :]]

    # Down cells
    if len(model.fix_arch_down_index.keys()) > 0:
        for key, value_lst in model.fix_arch_down_index.items():
            model.alphas_down.data[key, :] = value_lst[1]
    
    sort_log_alpha_down = torch.topk(F.softmax(model.alphas_down.data, dim=-1), 2)
    argmax_index_down = (sort_log_alpha_down[0][:, 0] - sort_log_alpha_down[0][:, 1] >= 0.3)

    for id in range(argmax_index_down.size(0)):
        if argmax_index_down[id] == 1 and id not in model.fix_arch_down_index.keys():
            model.fix_arch_down_index[id] = [sort_log_alpha_down[1][id, 0].item(),
                                             model.alphas_down.detach().clone()[id, :]]

    # Up cells
    if len(model.fix_arch_up_index.keys()) > 0:
        for key, value_lst in model.fix_arch_up_index.items():
            model.alphas_up.data[key, :] = value_lst[1]
    
    sort_log_alpha_up = torch.topk(F.softmax(model.alphas_up.data, dim=-1), 2)
    argmax_index_up = (sort_log_alpha_up[0][:, 0] - sort_log_alpha_up[0][:, 1] >= 0.3)

    for id in range(argmax_index_up.size(0)):
        if argmax_index_up[id] == 1 and id not in model.fix_arch_up_index.keys():
            model.fix_arch_up_index[id] = [sort_log_alpha_up[1][id, 0].item(),
                                           model.alphas_up.detach().clone()[id, :]]


def _log_architecture_decisions(model, logger, writer, epoch):
    """Log current architecture decisions"""
    with torch.no_grad():
        # Transformer connections
        if hasattr(model, 'alphas_transformer_connections'):
            conn_probs = F.softmax(model.alphas_transformer_connections, dim=-1)
            probs_on = conn_probs[:, 1].tolist()
            on_count = sum(1 for p in probs_on if p > 0.5)
            logger.info(f"  Transformer Connections: {on_count}/{len(probs_on)} ON")
            logger.info(f"    Probabilities: {[f'{p:.3f}' for p in probs_on]}")
        
        # Transformer configurations
        if hasattr(model, 'alphas_transformer_configs'):
            config_probs = F.softmax(model.alphas_transformer_configs, dim=-1)
            num_connections = config_probs.shape[0]
            logger.info(f"  Transformer Configurations:")
            for conn_idx in range(num_connections):
                probs = config_probs[conn_idx]
                top_config_idx = torch.argmax(probs).item()
                top_prob = probs[top_config_idx].item()
                top_config = TRANSFORMER_CONFIG_CHOICES[top_config_idx]
                complexity = TRANSFORMER_COMPLEXITY_LOOKUP[top_config_idx]
                logger.info(f"    Connection {conn_idx}: Config {top_config_idx} (prob={top_prob:.3f})")
                logger.info(f"      d_model={top_config['d_model']}, n_head={top_config['n_head']}, "
                           f"expansion={top_config['expansion']}")
                logger.info(f"      FLOPs={complexity['flops']:.2f}M, Params={complexity['params']:.2f}M")


def _print_final_architecture(model, logger):
    """Print final discovered architecture"""
    logger.info("\n" + "="*80)
    logger.info("FINAL DISCOVERED ARCHITECTURE")
    logger.info("="*80)
    
    # Transformer connections
    if hasattr(model, 'alphas_transformer_connections'):
        weight_transformer = F.softmax(model.alphas_transformer_connections, dim=-1).data.cpu().numpy()
        logger.info("Transformer Connections (ON probability):")
        for i, prob_on in enumerate(weight_transformer[:, 1]):
            status = "[ON]" if prob_on > 0.5 else "[OFF]"
            logger.info(f"  Connection {i}: {prob_on:.4f} {status}")
        total_on = sum(1 for p in weight_transformer[:, 1] if p > 0.5)
        logger.info(f"Total activated: {total_on}/{len(weight_transformer)}")
    
    # Transformer configurations
    if hasattr(model, 'alphas_transformer_configs'):
        config_weights = F.softmax(model.alphas_transformer_configs, dim=-1).data.cpu().numpy()
        logger.info("\nTransformer Configurations:")
        for conn_idx, probs in enumerate(config_weights):
            top_idx = np.argmax(probs)
            top_prob = probs[top_idx]
            config = TRANSFORMER_CONFIG_CHOICES[top_idx]
            complexity = TRANSFORMER_COMPLEXITY_LOOKUP[top_idx]
            logger.info(f"  Connection {conn_idx}:")
            logger.info(f"    Config: d_model={config['d_model']}, n_head={config['n_head']}, "
                       f"expansion={config['expansion']}")
            logger.info(f"    Complexity: FLOPs={complexity['flops']:.2f}M, Params={complexity['params']:.2f}M")
            logger.info(f"    Confidence: {top_prob:.4f}")
    
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet series Search with Pareto V2')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='Nas_Search_Unet', help='Model name')
    parser.add_argument('--note', type=str, default='pareto_v2', help="folder name note")
    parser.add_argument('--dataset', type=str, default='cvc', help='Dataset name')
    parser.add_argument('--dataset_root', type=str,
                        default='/mnt/data/KHTN2023/research25/hct-netm/datasets/cvc',
                        help='Dataset root path')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop size")
    parser.add_argument('--epochs', type=int, default=50, help="search epochs")
    parser.add_argument('--train_batch', type=int, default=2, help="train batch size")
    parser.add_argument('--val_batch', type=int, default=2, help="val batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader num workers")
    parser.add_argument('--train_portion', type=float, default=0.5, help="train/val split ratio")

    # Network architecture
    parser.add_argument('--num_classes', type=int, default=1, help="output feature channel")
    parser.add_argument('--input_c', type=int, default=3, help="input img channel")
    parser.add_argument('--init_channel', type=int, default=16, help="init channel for first level")
    parser.add_argument('--meta_node_num', type=int, default=4, help="meta nodes")
    parser.add_argument('--layers', type=int, default=9, help="number of layers")
    parser.add_argument('--use_sharing', type=bool, default=True, help="share operations")
    parser.add_argument('--depth', type=int, default=4, help="UNet depth")
    parser.add_argument('--double_down_channel', action='store_true', default=True)
    parser.add_argument('--dropout_prob', type=float, default=0, help="dropout prob")
    parser.add_argument('--use_softmax_head', type=bool, default=False)

    # Training settings
    parser.add_argument('--init_weight_type', type=str, default="kaiming", help="weight init")
    parser.add_argument('--arch_after', type=int, default=5, help="epochs before arch training")
    parser.add_argument('--infer_epoch', type=int, default=4, help="validation frequency")
    parser.add_argument('--compute_freq', type=int, default=40, help="compute frequency")
    parser.add_argument('--gpus', type=int, default=1, help="number of GPUs")
    parser.add_argument('--grad_clip', type=int, default=1, help="gradient clipping")
    parser.add_argument('--manualSeed', type=int, default=100, help="manual seed")

    # Loss and optimizer
    parser.add_argument('--loss', type=str, choices=['bcedice', 'bce', 'bcelog', 'dice', 'softdice'],
                        default="bcedice", help="loss function")
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--deepsupervision', action='store_true', default=True)
    parser.add_argument('--lr', type=float, default=0.025, help="learning rate")
    parser.add_argument('--lr_min', type=float, default=1e-5, help="min learning rate")
    parser.add_argument('--weight_decay', type=float, default=3e-4, help="weight decay")
    parser.add_argument('--arch_lr', type=float, default=1e-3, help="architecture lr")
    parser.add_argument('--arch_weight_decay', type=float, default=0, help="arch weight decay")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    
    # Resume
    parser.add_argument('--resume', type=str, default='None', help="resume checkpoint path")
    
    # DSNAS
    parser.add_argument('--early_fix_arch', action='store_true', default=True)
    parser.add_argument('--gen_max_child', action='store_true', default=True)
    parser.add_argument('--gen_max_child_flag', action='store_true', default=False)
    parser.add_argument('--random_sample', action='store_true', default=False)
    
    # Transformer settings
    parser.add_argument('--transformer_warmup_epochs', type=int, default=5,
                        help='warmup epochs for transformer')
    parser.add_argument('--transformer_init_bias', type=float, default=0.3,
                        help='initial bias for transformer connections')
    
    #  CÁCH 3: Positional bias parameters
    parser.add_argument('--positional_init_scale', type=float, default=0.1,
                        help='Scale factor ε for positional bias in alpha initialization (Cách 3)')
    parser.add_argument('--positional_bias_factor', type=float, default=0.05,
                        help='Scale factor γ for positional bias in complexity cost (Cách 3)')
    
    #  CÁCH 2: Diversity loss type
    parser.add_argument('--diversity_loss_type', type=str, default='variance',
                        choices=['indecision', 'variance', 'repulsion'],
                        help='Type of diversity loss: indecision (original), variance (Cách 2a), repulsion (Cách 2b)')
    
    #  CÁCH 4: Gumbel-Softmax parameters
    parser.add_argument('--use_gumbel_softmax', action='store_true', default=False,
                        help='Use Gumbel-Softmax for discrete sampling (Cách 4)')
    parser.add_argument('--gumbel_temperature', type=float, default=1.0,
                        help='Temperature for Gumbel-Softmax (lower = more discrete)')
    parser.add_argument('--gumbel_anneal', action='store_true', default=False,
                        help='Anneal Gumbel temperature during training')
    parser.add_argument('--gumbel_temp_min', type=float, default=0.5,
                        help='Minimum temperature for Gumbel annealing')
    
    # Pareto optimization weights (UPDATED for stronger learning)
    parser.add_argument('--pareto_weight_dice', type=float, default=0.3,
                        help='Weight for dice loss in Pareto optimization (30%)')
    parser.add_argument('--pareto_weight_complexity', type=float, default=0.3,
                        help='Weight for complexity loss (30% - encourage efficiency)')
    parser.add_argument('--pareto_weight_connection', type=float, default=0.4,
                        help='Weight for connection differentiation loss (40% - HIGHEST to force diverse decisions)')
    parser.add_argument('--pareto_scalarization', type=str, default='weighted_sum',
                        choices=['weighted_sum', 'tchebycheff', 'augmented_tchebycheff'],
                        help='Pareto scalarization method')
    
    args = parser.parse_args()
    main(args)