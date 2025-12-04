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

from genotypes import CellLinkDownPos, CellLinkUpPos, CellPos
from nas_model import get_models

sys.path.append('../')
from datasets import get_dataloder, datasets_dict
from utils import save_checkpoint, calc_parameters_count, get_logger, get_gpus_memory_info
from utils import BinaryIndicatorsMetric, AverageMeter
from utils import BCEDiceLoss, SoftDiceLoss, DiceLoss
from utils import LRScheduler


def pareto_dominates(obj1, obj2):
    """
    Check if obj1 Pareto dominates obj2.
    For minimization: obj1 dominates obj2 if obj1 is better or equal in all objectives 
    and strictly better in at least one.
    
    Args:
        obj1: tuple of objectives (can be 2D or 4D)
        obj2: tuple of objectives (can be 2D or 4D)
    
    Returns:
        True if obj1 dominates obj2
    """
    # Note: We minimize all objectives (including dice_loss = 1 - Dice)
    better_or_equal = all(o1 <= o2 for o1, o2 in zip(obj1, obj2))
    strictly_better = any(o1 < o2 for o1, o2 in zip(obj1, obj2))
    
    return better_or_equal and strictly_better


def aggregate_group_objective(dice_loss, entropy_loss):
    """
    Aggregate Group 1 (Accuracy) objectives into a single scalar.
    Uses weighted sum: more weight on dice (primary metric).
    
    Args:
        dice_loss: Dice loss value
        entropy_loss: Entropy loss value
        
    Returns:
        aggregated_accuracy: Single scalar for Group 1
    """
    # Weight: 70% Dice (primary), 30% Entropy (decisiveness)
    return 0.7 * dice_loss + 0.3 * entropy_loss


def aggregate_complexity_objective(complexity_loss, transformer_loss):
    """
    Aggregate Group 2 (Complexity) objectives into a single scalar.
    Uses weighted sum: balanced between operation cost and transformer decisions.
    
    Args:
        complexity_loss: Operation complexity loss
        transformer_loss: Transformer connection loss (negative reward)
        
    Returns:
        aggregated_complexity: Single scalar for Group 2
    """
    # Weight: 20% Operation complexity, 80% Transformer decisions
    return 0.2 * complexity_loss + 0.8 * transformer_loss


def update_pareto_front(pareto_front, new_solution):
    """
    Update Pareto front with a new solution.
    
    Args:
        pareto_front: list of (objectives_tuple, architecture_dict) 
        new_solution: (objectives_tuple, architecture_dict)
    
    Returns:
        updated pareto_front
    """
    new_obj, new_arch = new_solution
    
    # Remove solutions dominated by new solution
    pareto_front = [(obj, arch) for obj, arch in pareto_front 
                    if not pareto_dominates(new_obj, obj)]
    
    # Add new solution if it's not dominated by any existing solution
    if not any(pareto_dominates(obj, new_obj) for obj, _ in pareto_front):
        pareto_front.append(new_solution)
    
    return pareto_front


def hierarchical_pareto_optimization(dice_loss, entropy_loss, complexity_loss, transformer_loss, 
                                     group1_pareto_front, group2_pareto_front, top_level_pareto_front,
                                     architecture):
    """
    Hierarchical Pareto Optimization with 2 levels:
    
    Level 1 (Within-Group Pareto):
        - Group 1 Pareto: dice_loss vs entropy_loss
        - Group 2 Pareto: complexity_loss vs transformer_loss
    
    Level 2 (Top-Level Pareto):
        - Aggregate Group 1: aggregated_accuracy
        - Aggregate Group 2: aggregated_complexity
        - Pareto: aggregated_accuracy vs aggregated_complexity
    
    Args:
        dice_loss, entropy_loss, complexity_loss, transformer_loss: individual objectives (scalars)
        group1_pareto_front: Pareto front for Group 1 (accuracy)
        group2_pareto_front: Pareto front for Group 2 (complexity)
        top_level_pareto_front: Top-level Pareto front (2D)
        architecture: current architecture snapshot
        
    Returns:
        tuple: (updated_group1_front, updated_group2_front, updated_top_level_front)
    """
    # === LEVEL 1: Within-Group Pareto Optimization ===
    
    # Group 1: Accuracy (dice vs entropy)
    group1_objectives = (dice_loss, entropy_loss)
    group1_solution = (group1_objectives, architecture)
    group1_pareto_front = update_pareto_front(group1_pareto_front, group1_solution)
    
    # Group 2: Complexity (complexity vs transformer)
    group2_objectives = (complexity_loss, transformer_loss)
    group2_solution = (group2_objectives, architecture)
    group2_pareto_front = update_pareto_front(group2_pareto_front, group2_solution)
    
    # === LEVEL 2: Top-Level Pareto (Group aggregates) ===
    
    # Aggregate each group into single scalar
    aggregated_accuracy = aggregate_group_objective(dice_loss, entropy_loss)
    aggregated_complexity = aggregate_complexity_objective(complexity_loss, transformer_loss)
    
    # Top-level 2D Pareto optimization
    top_level_objectives = (aggregated_accuracy, aggregated_complexity)
    top_level_solution = (top_level_objectives, architecture)
    top_level_pareto_front = update_pareto_front(top_level_pareto_front, top_level_solution)
    
    return group1_pareto_front, group2_pareto_front, top_level_pareto_front


def get_architecture_snapshot(model):
    """
    Get current architecture parameters as a snapshot.
    
    Args:
        model: NAS model
        
    Returns:
        dict with architecture parameters
    """
    arch_dict = {
        'alphas_normal': model.alphas_normal.detach().cpu().clone(),
        'alphas_down': model.alphas_down.detach().cpu().clone(),
        'alphas_up': model.alphas_up.detach().cpu().clone(),
        'alphas_network': model.alphas_network.detach().cpu().clone(),
    }
    
    if hasattr(model, 'alphas_transformer_connections'):
        arch_dict['alphas_transformer_connections'] = model.alphas_transformer_connections.detach().cpu().clone()
    
    return arch_dict


def compute_multi_objective_losses(model, criterion, preds, target, args, epoch=None, total_epochs=None):
    """
    Compute multiple objectives separately for Pareto optimization.
    Focus on DICE and JACCARD metrics for medical segmentation.
    
    Args:
        model: NAS model
        criterion: loss function (BCEDice)
        preds: predictions
        target: ground truth
        args: arguments
        epoch: current epoch (for progressive scheduling)
        total_epochs: total epochs (for progressive scheduling)
        
    Returns:
        tuple of (dice_loss, complexity_loss, entropy_loss, transformer_loss)
    """
    # Objective 1: DICE-focused segmentation loss
    # Extract Dice component from BCEDice loss for optimization
    if args.deepsupervision:
        # Compute Dice loss for all predictions
        dice_losses = []
        for pred in preds:
            # Compute Dice coefficient: 2*|X∩Y| / (|X| + |Y|)
            pred_sigmoid = torch.sigmoid(pred)
            intersection = (pred_sigmoid * target).sum()
            union = pred_sigmoid.sum() + target.sum()
            dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
            dice_loss = 1.0 - dice  # Convert to loss (minimize)
            dice_losses.append(dice_loss)
        dice_objective = sum(dice_losses) / len(dice_losses)
    else:
        # Single output Dice loss
        pred_sigmoid = torch.sigmoid(preds[-1])
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
        dice_objective = 1.0 - dice
    
    # Objective 2: Model complexity (weighted by operation costs)
    complexity_loss = compute_complexity_loss(model, args)
    
    # Objective 3: Architecture entropy (decisiveness) with progressive scheduling
    entropy_loss = compute_architecture_complexity_loss(model, args, epoch, total_epochs)
    
    # Objective 4: Transformer connection usage - DICE-BASED REWARD
    if hasattr(model, 'alphas_transformer_connections'):
        transformer_loss, updated_ema = compute_transformer_connection_loss(
            model, args, dice_objective, getattr(model, '_ema_dice_loss', None), epoch, total_epochs
        )
        model._ema_dice_loss = updated_ema
    else:
        transformer_loss = torch.tensor(0.0, device=dice_objective.device)
    
    return (dice_objective.item(), 
            complexity_loss.item(), 
            entropy_loss.item(), 
            transformer_loss.item())


def select_loss_from_pareto(pareto_front, strategy='random'):
    """
    Select a solution from Pareto front to guide gradient update.
    Works with both 2D (top-level) and 4D (within-group) Pareto fronts.
    
    Args:
        pareto_front: list of (objectives, architecture)
        strategy: 'random', 'balanced', or 'dice_focused'
        
    Returns:
        selected objectives tuple
    """
    if not pareto_front:
        return None
    
    if strategy == 'random':
        # Randomly select from Pareto front
        idx = np.random.randint(len(pareto_front))
        return pareto_front[idx][0]
    
    elif strategy == 'balanced':
        # Select solution with best average normalized rank
        objectives_array = np.array([obj for obj, _ in pareto_front])
        
        # Normalize each objective to [0, 1]
        min_vals = objectives_array.min(axis=0)
        max_vals = objectives_array.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # Avoid division by zero
        
        normalized = (objectives_array - min_vals) / ranges
        avg_scores = normalized.mean(axis=1)
        
        best_idx = avg_scores.argmin()
        return pareto_front[best_idx][0]
    
    elif strategy == 'dice_focused':
        # Select solution with best first objective (works for both 2D and 4D)
        # For top-level: aggregated_accuracy
        # For group1: dice_loss
        best_idx = min(range(len(pareto_front)), 
                      key=lambda i: pareto_front[i][0][0])
        return pareto_front[best_idx][0]
    
    return pareto_front[0][0]


def compute_pareto_guided_loss(objectives, dice_loss, complexity_loss, 
                                entropy_loss, transformer_loss, args):
    """
    Compute weighted loss based on target objectives from TOP-LEVEL Pareto front.
    For hierarchical Pareto: objectives = (aggregated_accuracy, aggregated_complexity)
    
    Args:
        objectives: 2D tuple from top-level Pareto (aggregated_accuracy, aggregated_complexity)
        dice_loss, complexity_loss, entropy_loss, transformer_loss: current losses (tensors with gradients)
        args: arguments
        
    Returns:
        weighted combined loss preserving hierarchical structure
    """
    if len(objectives) == 2:
        # === HIERARCHICAL MODE: Top-level 2D Pareto ===
        target_aggregated_accuracy, target_aggregated_complexity = objectives
        
        # Recompute current aggregated objectives (with gradients!)
        current_aggregated_accuracy = aggregate_group_objective(
            dice_loss.detach().item(), 
            entropy_loss.detach().item()
        )
        current_aggregated_complexity = aggregate_complexity_objective(
            complexity_loss.detach().item(), 
            transformer_loss.detach().item()
        )
        
        # Compute weights based on distance to target
        weights = []
        for curr, target in [(current_aggregated_accuracy, target_aggregated_accuracy),
                              (current_aggregated_complexity, target_aggregated_complexity)]:
            if target > 0:
                weight = max(0.1, curr / target)
            else:
                weight = 1.0
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Combine losses using hierarchical aggregation (preserves gradient flow)
        group1_loss = 0.7 * dice_loss + 0.3 * entropy_loss  # Group 1: Accuracy
        group2_loss = 0.5 * complexity_loss + 0.5 * transformer_loss  # Group 2: Complexity
        
        loss = weights[0] * group1_loss + weights[1] * group2_loss
        
    else:
        # === FALLBACK: 4D Pareto (for backward compatibility) ===
        target_dice, target_comp, target_ent, target_trans = objectives
        
        weights = []
        current_objs = [dice_loss, complexity_loss, entropy_loss, transformer_loss]
        
        for curr_tensor, target in zip(current_objs, objectives):
            curr = curr_tensor.detach().item()
            if target > 0:
                weight = max(0.1, curr / target)
            else:
                weight = 1.0
            weights.append(weight)
        
        # Boost transformer weight
        weights[3] *= 4.0
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        loss = (weights[0] * dice_loss + 
                weights[1] * complexity_loss + 
                weights[2] * entropy_loss + 
                weights[3] * transformer_loss)
    
    return loss


def compute_complexity_loss(model, args):
    """
    Compute weighted complexity loss based on actual operation costs.
    Different operations have different computational costs.
    
    Args:
        model: The NAS model with architecture parameters
        args: Arguments containing complexity_weight
    
    Returns:
        complexity_loss: Weighted sum based on operation costs
    """
    if not hasattr(model, 'alphas_normal'):
        return torch.tensor(0.0)
    
    # Define operation complexity costs (relative to simplest operation)
    op_costs_normal = {
        'none': 0.0,
        'identity': 0.1,
        'cweight': 0.2,
        'conv': 0.5,
        'shuffle_conv': 0.4,
        'dep_conv': 0.3,
        'dil_conv': 0.6,
        'Dynamic_convOPS': 1.0,  # Most expensive
    }
    
    op_costs_down = {
        'avg_pool': 0.2,
        'max_pool': 0.2,
        'down_cweight': 0.3,
        'down_conv': 0.6,
        'down_dep_conv': 0.4,
        'down_dil_conv': 0.7,
        'down_Dynamic_convOPS': 1.2,
    }
    
    op_costs_up = {
        'up_cweight': 0.3,
        'up_conv': 0.7,
        'up_dep_conv': 0.5,
        'up_dil_conv': 0.8,
        'up_Dynamic_convOPS': 1.3,
    }
    
    device = model.alphas_normal.device
    total_cost = 0
    num_edges = 0
    
    # Normal cell operations
    if hasattr(model, 'alphas_normal'):
        probs = F.softmax(model.alphas_normal, dim=-1)
        costs = torch.tensor([op_costs_normal.get(op, 0.5) for op in model.CellPos], 
                            device=device, dtype=torch.float32)
        edge_costs = (probs * costs).sum(dim=-1)
        total_cost += edge_costs.sum()
        num_edges += model.alphas_normal.size(0)
    
    # Down-sampling operations
    if hasattr(model, 'alphas_down'):
        probs = F.softmax(model.alphas_down, dim=-1)
        costs = torch.tensor([op_costs_down.get(op, 0.5) for op in model.CellLinkDownPos], 
                            device=device, dtype=torch.float32)
        edge_costs = (probs * costs).sum(dim=-1)
        total_cost += edge_costs.sum()
        num_edges += model.alphas_down.size(0)
    
    # Up-sampling operations
    if hasattr(model, 'alphas_up'):
        probs = F.softmax(model.alphas_up, dim=-1)
        costs = torch.tensor([op_costs_up.get(op, 0.5) for op in model.CellLinkUpPos], 
                            device=device, dtype=torch.float32)
        edge_costs = (probs * costs).sum(dim=-1)
        total_cost += edge_costs.sum()
        num_edges += model.alphas_up.size(0)
    
    # Average weighted complexity
    complexity = total_cost / num_edges if num_edges > 0 else torch.tensor(0.0, device=device)
    
    return complexity


def compute_architecture_complexity_loss(model, args, epoch=None, total_epochs=None):
    """
    Compute architecture-level complexity loss based on entropy of architecture parameters.
    Higher entropy means the architecture is more uncertain/complex.
    Lower entropy (more decisive choices) is preferred.
    
    Uses progressive penalty scheduling to force convergence over time.
    
    Args:
        model: The NAS model with architecture parameters
        args: Arguments containing arch_complexity_weight
        epoch: Current epoch (for progressive scheduling)
        total_epochs: Total number of epochs (for progressive scheduling)
    
    Returns:
        arch_complexity_loss: Entropy-based complexity penalty
    """
    def compute_entropy(alphas):
        """Compute entropy of softmax distribution"""
        probs = F.softmax(alphas, dim=-1)
        log_probs = F.log_softmax(alphas, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return torch.mean(entropy)
    
    total_entropy = 0.0
    count = 0
    
    # Compute entropy for all architecture parameters
    if hasattr(model, 'alphas_normal'):
        total_entropy += compute_entropy(model.alphas_normal)
        count += 1
    
    if hasattr(model, 'alphas_down'):
        total_entropy += compute_entropy(model.alphas_down)
        count += 1
    
    if hasattr(model, 'alphas_up'):
        total_entropy += compute_entropy(model.alphas_up)
        count += 1
    
    if hasattr(model, 'alphas_network'):
        total_entropy += compute_entropy(model.alphas_network)
        count += 1
    
    # Average entropy across all architecture parameters
    avg_entropy = total_entropy / count if count > 0 else torch.tensor(0.0)
    
    # Progressive penalty scheduling: increase penalty over time
    if epoch is not None and total_epochs is not None and epoch >= args.arch_after:
        progress = (epoch - args.arch_after) / (total_epochs - args.arch_after)
        # Target entropy: we want entropy to decrease to ~0.5 (decisive distribution)
        target_entropy = 0.5
        
        # Penalty increases with progress (1.0 → 4.0)
        penalty_multiplier = 1.0 + 3.0 * progress
        
        # Only penalize if above target
        entropy_penalty = torch.relu(avg_entropy - target_entropy) * penalty_multiplier
        return entropy_penalty
    
    return avg_entropy


def compute_transformer_connection_loss(model, args, dice_loss_tensor, ema_dice_loss, epoch=None, total_epochs=None, pareto_target_dice=None):
    """
    Compute loss for transformer connections with HIERARCHICAL CHAIN RULE gradient flow.
    
    *** KEY IMPROVEMENT: CASCADING GRADIENT FLOW (CHAIN RULE) ***
    - Gradient flows through layers SEQUENTIALLY: Output → Layer N → Layer N-1 → ... → Layer 1
    - Each layer's gradient depends on DOWNSTREAM layers' activations (chain rule)
    - Mimics true backpropagation: ∂Loss/∂Layer_i = ∂Loss/∂Output × ∂Output/∂Layer_j × ... × ∂Layer_j/∂Layer_i
    
    Design Philosophy:
    1. **Cascading Gradient**: Layer closer to output receives stronger, more direct gradient
    2. **Dependency Chain**: Upstream layers modulated by downstream layer probabilities
    3. **Hierarchical Reward**: Deep layers influence shallow layers through multiplicative gating
    4. **Natural Exploration**: Model learns layer importance hierarchy automatically
    
    Connection Order (for layers=9):
    - Connection 0: Layer 2 ↔ Layer 9 (DEEPEST, closest to output, index=0)
    - Connection 1: Layer 3 ↔ Layer 8  
    - Connection 2: Layer 4 ↔ Layer 7
    - Connection 3: Layer 5 ↔ Layer 6 (SHALLOWEST, furthest from output, index=3)
    
    Chain Rule Example:
    - Connection 0: gradient = base_gradient × 1.0 (direct from output)
    - Connection 1: gradient = base_gradient × prob_on[0] (gated by connection 0)
    - Connection 2: gradient = base_gradient × prob_on[0] × prob_on[1] (gated by 0 & 1)
    - Connection 3: gradient = base_gradient × prob_on[0] × prob_on[1] × prob_on[2] (gated by all)
    
    Args:
        model: The NAS model with transformer connection parameters
        args: Arguments containing transformer_connection_weight
        dice_loss_tensor: Current dice loss tensor (1 - Dice score) WITH GRADIENT!
        ema_dice_loss: EMA of dice loss (scalar, no gradient, for tracking only)
        epoch: Current epoch (for warmup scheduling)
        total_epochs: Total epochs (for warmup scheduling)
        pareto_target_dice: Target Dice loss from Pareto front's best solution (optional)
    
    Returns:
        tuple: (transformer_loss, updated_ema_dice_loss)
    """
    if not hasattr(model, 'alphas_transformer_connections'):
        return torch.tensor(0.0).to(args.device), ema_dice_loss
    
    # Get probabilities for transformer connections (ON vs OFF)
    alphas_transformer = F.softmax(model.alphas_transformer_connections, dim=-1)
    probs_on = alphas_transformer[:, 1]  # ON probabilities [num_connections]
    num_connections = probs_on.shape[0]
    
    # === UPDATE EMA (detached for tracking only, not used in loss) ===
    dice_loss_value = dice_loss_tensor.detach().item()
    if ema_dice_loss is None:
        new_ema_dice_loss = dice_loss_value
    else:
        new_ema_dice_loss = 0.9 * ema_dice_loss + 0.1 * dice_loss_value
    
    # === COMPUTE BASE PERFORMANCE SCORE (shared by all connections) ===
    if pareto_target_dice is not None:
        # Use Pareto front's best Dice as target (convert to tensor, no gradient needed)
        target_tensor = torch.tensor(pareto_target_dice, 
                                     device=dice_loss_tensor.device,
                                     dtype=dice_loss_tensor.dtype)
        # *** KEY: Compute improvement WITH gradient from dice_loss_tensor ***
        dice_improvement = target_tensor - dice_loss_tensor  # KEEPS GRADIENT!
        base_performance_score = torch.sigmoid(dice_improvement * 10.0)  # KEEPS GRADIENT!
    else:
        # Fallback: compare to EMA (tensor without gradient)
        ema_tensor = torch.tensor(new_ema_dice_loss, 
                                  device=dice_loss_tensor.device,
                                  dtype=dice_loss_tensor.dtype)
        # *** KEY: Improvement has gradient through dice_loss_tensor ***
        dice_improvement = ema_tensor - dice_loss_tensor  # KEEPS GRADIENT!
        base_performance_score = torch.sigmoid(dice_improvement * 10.0)  # KEEPS GRADIENT!
    
    # === HIERARCHICAL CHAIN RULE: Cascading Gradient Flow ===
    # Connection 0 (closest to output) receives FULL gradient
    # Each subsequent connection receives gradient MODULATED by all downstream connections
    
    cascading_rewards = []
    for i in range(num_connections):
        # Compute chain rule factor: product of all downstream probabilities
        # Connections closer to output (lower index) have fewer/no dependencies
        if i == 0:
            # Deepest connection (closest to output): Direct gradient, no modulation
            chain_factor = torch.tensor(1.0, device=probs_on.device, dtype=probs_on.dtype)
        else:
            # Upstream connections: Gradient gated by ALL downstream connections
            # This implements chain rule: ∂Loss/∂Connection_i = ∂Loss/∂Output × ∏(∂Output/∂Connection_j)
            downstream_probs = probs_on[:i]  # All connections closer to output (indices 0 to i-1)
            
            # Multiplicative gating: If downstream connections are OFF, upstream gradient is weak
            # Add small epsilon (0.1) to prevent complete gradient vanishing
            chain_factor = torch.prod(downstream_probs + 0.1)  # KEEPS GRADIENT through probs_on!
            
            # Alternative softer dependency (uncomment to try):
            # chain_factor = torch.mean(downstream_probs + 0.1)
        
        # Each connection receives reward scaled by its position in the chain
        # WITH GRADIENT FLOW through both base_performance_score AND chain_factor!
        connection_reward = base_performance_score * chain_factor * probs_on[i]
        cascading_rewards.append(connection_reward)
    
    # Stack into tensor for efficient computation
    cascading_rewards = torch.stack(cascading_rewards)  # [num_connections], all with gradient!
    
    # === STRONG CASCADING REWARD ===
    # Negative loss = reward for turning ON connections
    # Connections closer to output naturally receive stronger gradient due to chain_factor
    dice_based_reward = -5.0 * cascading_rewards.mean()
    
    # === COMPLEXITY COST: Light penalty, same for all connections ===
    expected_num_on = probs_on.sum()
    complexity_cost = expected_num_on * 0.1
    
    # === WARMUP PHASE: Even lighter penalty during exploration ===
    warmup_scale = 1.0
    if epoch is not None and hasattr(args, 'transformer_warmup_epochs'):
        if epoch < args.transformer_warmup_epochs:
            warmup_scale = 0.05  # Very light during warmup
    
    # === DECISIVENESS LOSS: Weak, encourages convergence ===
    if epoch is not None and hasattr(args, 'transformer_warmup_epochs'):
        if epoch >= args.transformer_warmup_epochs:
            uncertainty = 1.0 - torch.abs(probs_on - 0.5) * 2.0
            decisiveness_loss = 0.05 * uncertainty.mean()
        else:
            decisiveness_loss = torch.tensor(0.0, device=probs_on.device)
    else:
        uncertainty = 1.0 - torch.abs(probs_on - 0.5) * 2.0
        decisiveness_loss = 0.05 * uncertainty.mean()
    
    # === Combined loss with HIERARCHICAL GRADIENT (CHAIN RULE) ===
    # Priority: Cascading Dice reward >>> Complexity >> Decisiveness
    transformer_loss = (
        dice_based_reward +  # CASCADING reward with chain rule gradient flow!
        warmup_scale * complexity_cost +
        decisiveness_loss
    )
    
    return transformer_loss, new_ema_dice_loss


def main(args):
    ############    init config ################
    #################### init logger ###################################
    log_dir = './search_exp/' + '/{}'.format(args.model) + \
              '/{}'.format(args.dataset) + '/{}_{}'.format(time.strftime('%Y%m%d-%H%M%S'), args.note)

    logger = get_logger(log_dir)
    print('RUNDIR: {}'.format(log_dir))
    logger.info('{}-Search'.format(args.model))
    args.save_path = log_dir
    args.save_tbx_log = args.save_path + '/tbx_log'
    writer = SummaryWriter(args.save_tbx_log)
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
    
    # Initialize transformer switches based on number of layers
    # For transformer configurations, we need switches for each transformer block
    num_transformer_configs = 4  # Number of transformer configuration options
    for i in range(args.layers):
        switches_transformer.append([True for j in range(num_transformer_configs)])

    original_train_batch = args.train_batch
    original_val_batch = args.val_batch

    #############################select model################################
    # Model can be dynamically configured with different number of layers
    # For UNet-style architectures: layers can be 3, 5, 7, 9, etc. (odd numbers)
    # Number of transformer connections will be: (layers - 1) / 2
    args.model = "UnetLayer{}".format(args.layers)  # Use layers from args
    sp_train_batch = original_train_batch
    sp_val_batch = original_val_batch
    sp_lr = args.lr
    sp_epoch = args.epochs
    early_fix_arch = args.early_fix_arch
    gen_max_child_flag = args.gen_max_child_flag
    random_sample = args.random_sample
    
    logger.info(f"Building model with {args.layers} layers")
    logger.info(f"Expected transformer connections: {(args.layers - 1) // 2}")

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

    # === INITIALIZE TRANSFORMER ALPHAS WITH BIAS TOWARD ON ===
    if hasattr(model, 'alphas_transformer_connections') and hasattr(args, 'transformer_init_bias'):
        with torch.no_grad():
            # Initialize with bias: higher values for ON (index 1), lower for OFF (index 0)
            # Lower bias (e.g., 0.3) gives ~57% initial ON probability for more exploration
            # Higher bias (e.g., 2.0) gives ~88% initial ON probability
            bias = args.transformer_init_bias
            for i in range(len(model.alphas_transformer_connections)):
                model.alphas_transformer_connections[i, 0] = -bias  # OFF (lower)
                model.alphas_transformer_connections[i, 1] = bias   # ON (higher)
            
            # Verify initialization
            init_probs = F.softmax(model.alphas_transformer_connections, dim=-1)
            logger.info("=" * 80)
            logger.info("TRANSFORMER INITIALIZATION")
            logger.info("=" * 80)
            logger.info(f"Initialized {len(model.alphas_transformer_connections)} transformer connections")
            logger.info(f"Initial bias value: {bias} ({'low exploration' if bias > 1.0 else 'high exploration'})")
            logger.info(f"Initial ON probabilities: {[f'{p:.3f}' for p in init_probs[:, 1].tolist()]}")
            logger.info(f"Expected ON count: {sum(1 for p in init_probs[:, 1] if p > 0.5)}/{len(init_probs)}")
            logger.info(f"Transformer fixing: {'ENABLED' if getattr(args, 'fix_transformer_arch', False) else 'DISABLED (recommended)'}")
            if getattr(args, 'fix_transformer_arch', False):
                logger.info(f"  - Will fix after epoch: {getattr(args, 'transformer_min_epochs_before_fix', args.transformer_warmup_epochs + 10)}")
                logger.info(f"  - Fix margin: {getattr(args, 'transformer_fix_margin', 0.6)} (vs 0.3 for operations)")
            logger.info("=" * 80 + "\n")
    
    for v in model.parameters():
        if v.requires_grad:
            if v.grad is None:
                v.grad = torch.zeros_like(v)
    model.alphas_up.grad = torch.zeros_like(model.alphas_up)
    model.alphas_down.grad = torch.zeros_like(model.alphas_down)
    model.alphas_normal.grad = torch.zeros_like(model.alphas_normal)
    model.alphas_network.grad = torch.zeros_like(model.alphas_network)
    
    # Initialize gradient for transformer connection parameters
    if hasattr(model, 'alphas_transformer_connections'):
        model.alphas_transformer_connections.grad = torch.zeros_like(model.alphas_transformer_connections)

    wo_wd_params = []
    wo_wd_param_names = []
    network_params = []
    network_param_names = []
    # print("1",model.named_modules())
    for name, mod in model.named_modules():
        # print("1:",name,mod)
        if isinstance(mod, nn.BatchNorm2d):
            for key, value in mod.named_parameters():
                wo_wd_param_names.append(name + '.' + key)
            # print("pa:",wo_wd_param_names)

    for key, value in model.named_parameters():
        if "alphas" not in key:
            if value.requires_grad:
                if key in wo_wd_param_names:
                    wo_wd_params.append(value)  # 模块参数名字 权值
                else:
                    network_params.append(value)
                    network_param_names.append(key)  # 模块以外的参数


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
            #optimizer_cellarch.load_state_dict(checkpoint['optimizer_cellarch'])
            optimizer_arch.load_state_dict(checkpoint['optimizer_arch'])
            optimizer_weight.load_state_dict(checkpoint['optimizer_weight'])
            model_inner = model.module if hasattr(model, 'module') else model
            model_inner.load_alphas(checkpoint['alphas_dict'])
            model.load_state_dict(checkpoint['state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))

    # Initialize HIERARCHICAL Pareto fronts for multi-objective optimization
    group1_pareto_front = []  # Group 1: Accuracy (dice vs entropy)
    group2_pareto_front = []  # Group 2: Complexity (complexity vs transformer)
    top_level_pareto_front = []  # Top-level: aggregated_accuracy vs aggregated_complexity
    
    logger.info("="*80)
    logger.info("HIERARCHICAL PARETO OPTIMIZATION INITIALIZED")
    logger.info("="*80)
    logger.info("Level 1 - Within-Group Pareto:")
    logger.info("  Group 1 (Accuracy): dice_loss vs entropy_loss")
    logger.info("  Group 2 (Complexity): complexity_loss vs transformer_loss")
    logger.info("Level 2 - Top-Level Pareto:")
    logger.info("  aggregated_accuracy (0.7*dice + 0.3*entropy) vs")
    logger.info("  aggregated_complexity (0.5*complexity + 0.5*transformer)")
    logger.info("="*80 + "\n")
    
    # Initialize EMA for Dice loss (used for transformer connection optimization)
    ema_acc_loss = None  # Variable name kept for compatibility, but tracks Dice loss now
    
    # Set Pareto strategy: 'random', 'balanced', or 'dice_focused'
    if not hasattr(args, 'pareto_strategy'):
        args.pareto_strategy = 'balanced'
    logger.info(f"Using Pareto selection strategy: {args.pareto_strategy}")

    max_value=0
    for epoch in range(start_epoch, sp_epoch):

        logger.info('################Epoch: %d lr %e######################', epoch, scheduler.get_last_lr()[0])

        if args.early_fix_arch:
            # Fix architecture for cell operations
            if len(model.fix_arch_normal_index.keys()) > 0:
                for key, value_lst in model.fix_arch_normal_index.items():
                    model.alphas_normal.data[key, :] = value_lst[1]

            sort_log_alpha_normal = torch.topk(F.softmax(model.alphas_normal.data, dim=-1), 2)
            argmax_index_normal = (sort_log_alpha_normal[0][:, 0] - sort_log_alpha_normal[0][:, 1] >= 0.3)

            for id in range(argmax_index_normal.size(0)):
                if argmax_index_normal[id] == 1 and id not in model.fix_arch_normal_index.keys():
                    model.fix_arch_normal_index[id] = [sort_log_alpha_normal[1][id, 0].item(),
                                                       model.alphas_normal.detach().clone()[id, :]]

            if len(model.fix_arch_down_index.keys()) > 0:
                for key, value_lst in model.fix_arch_down_index.items():
                    model.alphas_down.data[key, :] = value_lst[1]
            sort_log_alpha_down = torch.topk(F.softmax(model.alphas_down.data, dim=-1), 2)
            argmax_index_down = (sort_log_alpha_down[0][:, 0] - sort_log_alpha_down[0][:, 1] >= 0.3)

            for id in range(argmax_index_down.size(0)):
                if argmax_index_down[id] == 1 and id not in model.fix_arch_down_index.keys():
                    model.fix_arch_down_index[id] = [sort_log_alpha_down[1][id, 0].item(),
                                                     model.alphas_down.detach().clone()[id, :]]

            if len(model.fix_arch_up_index.keys()) > 0:
                for key, value_lst in model.fix_arch_up_index.items():
                    model.alphas_up.data[key, :] = value_lst[1]
            sort_log_alpha_up = torch.topk(F.softmax(model.alphas_up.data, dim=-1), 2)
            argmax_index_up = (sort_log_alpha_up[0][:, 0] - sort_log_alpha_up[0][:, 1] >= 0.3)

            for id in range(argmax_index_up.size(0)):
                if argmax_index_up[id] == 1 and id not in model.fix_arch_up_index.keys():
                    model.fix_arch_up_index[id] = [sort_log_alpha_up[1][id, 0].item(),
                                                   model.alphas_up.detach().clone()[id, :]]
            
            # ============ TRANSFORMER FIXING LOGIC (SEPARATE & DELAYED) ============
            # Only fix transformers if:
            # 1. Explicitly enabled via flag (default: False)
            # 2. After sufficient learning epochs (warmup + learning time)
            # 3. With higher confidence margin (0.6 vs 0.3 for operations)
            
            TRANSFORMER_FIX_ENABLED = getattr(args, 'fix_transformer_arch', False)
            MIN_EPOCHS_BEFORE_FIX = getattr(args, 'transformer_min_epochs_before_fix', 
                                           args.transformer_warmup_epochs + 10)
            TRANSFORMER_FIX_MARGIN = getattr(args, 'transformer_fix_margin', 0.6)
            
            if not hasattr(model, 'fix_arch_transformer_index'):
                model.fix_arch_transformer_index = {}
            
            if TRANSFORMER_FIX_ENABLED and epoch >= MIN_EPOCHS_BEFORE_FIX:
                # Apply already-fixed values
                if len(model.fix_arch_transformer_index.keys()) > 0:
                    for key, value_lst in model.fix_arch_transformer_index.items():
                        model.alphas_transformer_connections.data[key, :] = value_lst[1]
                
                # Check for new fixable transformers with HIGHER margin
                sort_log_alpha_transformer = torch.topk(
                    F.softmax(model.alphas_transformer_connections.data, dim=-1), 2
                )
                argmax_index_transformer = (
                    sort_log_alpha_transformer[0][:, 0] - 
                    sort_log_alpha_transformer[0][:, 1] >= TRANSFORMER_FIX_MARGIN
                )
                
                for id in range(argmax_index_transformer.size(0)):
                    if argmax_index_transformer[id] == 1 and id not in model.fix_arch_transformer_index.keys():
                        decision_idx = sort_log_alpha_transformer[1][id, 0].item()
                        model.fix_arch_transformer_index[id] = [
                            decision_idx,
                            model.alphas_transformer_connections.detach().clone()[id, :]
                        ]
                        decision = 'ON' if decision_idx == 1 else 'OFF'
                        prob = sort_log_alpha_transformer[0][id, 0].item()
                        logger.info(f"Fixed transformer connection {id}: {decision} (prob={prob:.4f})")
            else:
                # Don't fix transformers yet - let them learn!
                if epoch % 5 == 0 and epoch >= args.arch_after:  # Log every 5 epochs after arch training starts
                    if not TRANSFORMER_FIX_ENABLED:
                        logger.info(f"Epoch {epoch}: Transformer fixing is DISABLED - transformers will learn freely")
                    else:
                        logger.info(f"Epoch {epoch}: Transformers still learning (will fix after epoch {MIN_EPOCHS_BEFORE_FIX})")
        
        if epoch < args.arch_after:
          weight_loss_avg, arch_loss_avg, complexity_loss_avg, arch_complexity_loss_avg, transformer_loss_avg, mr, ms, mp, mf, mjc, md, macc, epoch_time, group1_pareto_front, group2_pareto_front, top_level_pareto_front, ema_acc_loss = train(args, train_queue, val_queue, model, criterion, optimizer_weight, optimizer_arch, group1_pareto_front, group2_pareto_front, top_level_pareto_front, ema_acc_loss,
                                                                              train_arch=False, epoch=epoch, total_epochs=sp_epoch)
        else:
          weight_loss_avg, arch_loss_avg, complexity_loss_avg, arch_complexity_loss_avg, transformer_loss_avg, mr, ms, mp, mf, mjc, md, macc, epoch_time, group1_pareto_front, group2_pareto_front, top_level_pareto_front, ema_acc_loss = train(args, train_queue, val_queue, model, criterion, optimizer_weight, optimizer_arch, group1_pareto_front, group2_pareto_front, top_level_pareto_front, ema_acc_loss,
                                                                              train_arch=True, epoch=epoch, total_epochs=sp_epoch)

        logger.info("Epoch:{} WeightLoss:{:.3f}  ArchLoss:{:.3f}".format(epoch, weight_loss_avg, arch_loss_avg))
        logger.info("         Acc:{:.3f}   Dice:{:.3f}  Jc:{:.3f}".format(macc, md, mjc))
        if args.enable_complexity_loss:
            logger.info("         ComplexityLoss:{:.4f}  ArchComplexityLoss:{:.4f}  TransformerLoss:{:.4f}".format(
                complexity_loss_avg, arch_complexity_loss_avg, transformer_loss_avg))
        
        # Log architecture convergence metrics
        if epoch >= args.arch_after:
            with torch.no_grad():
                # Measure entropy (lower is better - more decisive)
                alphas_normal_probs = F.softmax(model.alphas_normal, dim=-1)
                alphas_normal_entropy = -(alphas_normal_probs * torch.log(alphas_normal_probs + 1e-8)).sum(dim=-1).mean().item()
                
                # Measure max probability (higher is better - more decisive)
                alphas_normal_max_prob = alphas_normal_probs.max(dim=-1)[0].mean().item()
                
                # Measure transformer decisiveness with WARMUP indicator
                if hasattr(model, 'alphas_transformer_connections'):
                    trans_probs = F.softmax(model.alphas_transformer_connections, dim=-1)
                    trans_on_probs = trans_probs[:, 1].tolist()
                    trans_decisiveness = torch.abs(trans_probs[:, 0] - trans_probs[:, 1]).mean().item()
                    trans_on_count = sum(1 for p in trans_on_probs if p > 0.5)
                    
                    # Check if transformers are fixed
                    fixed_count = len(model.fix_arch_transformer_index) if hasattr(model, 'fix_arch_transformer_index') else 0
                    
                    warmup_status = f" [WARMUP {epoch}/{args.transformer_warmup_epochs}]" if epoch < args.transformer_warmup_epochs else ""
                    fix_status = f" [FIXED: {fixed_count}/{len(trans_on_probs)}]" if fixed_count > 0 else " [LEARNING]"
                    
                    logger.info(f"         Entropy: {alphas_normal_entropy:.4f}, MaxProb: {alphas_normal_max_prob:.4f}")
                    logger.info(f"         TransDecision: {trans_decisiveness:.4f}, TransON: {[f'{p:.3f}' for p in trans_on_probs]}")
                    logger.info(f"         TransCount: {trans_on_count}/{len(trans_on_probs)} transformers ON{warmup_status}{fix_status}")
                    
                    # Log gradient flow for transformers
                    if hasattr(model.alphas_transformer_connections, 'grad') and model.alphas_transformer_connections.grad is not None:
                        grad_norm = model.alphas_transformer_connections.grad.norm().item()
                        logger.info(f"         TransGrad: {grad_norm:.6f} (gradient norm)")
                    
                    writer.add_scalar('Architecture/Entropy', alphas_normal_entropy, epoch)
                    writer.add_scalar('Architecture/MaxProb', alphas_normal_max_prob, epoch)
                    writer.add_scalar('Architecture/TransDecisiveness', trans_decisiveness, epoch)
                    writer.add_scalar('Architecture/TransCountON', trans_on_count, epoch)
                    writer.add_scalar('Architecture/TransFixedCount', fixed_count, epoch)
                else:
                    logger.info(f"         Entropy: {alphas_normal_entropy:.4f}, MaxProb: {alphas_normal_max_prob:.4f}")
                    writer.add_scalar('Architecture/Entropy', alphas_normal_entropy, epoch)
                    writer.add_scalar('Architecture/MaxProb', alphas_normal_max_prob, epoch)
        
        # Log HIERARCHICAL Pareto front information
        logger.info(f"         [HIERARCHICAL PARETO] Top-Level: {len(top_level_pareto_front)} solutions")
        logger.info(f"         [HIERARCHICAL PARETO] Group1 (Accuracy): {len(group1_pareto_front)} solutions")
        logger.info(f"         [HIERARCHICAL PARETO] Group2 (Complexity): {len(group2_pareto_front)} solutions")
        
        if len(top_level_pareto_front) > 0:
            best_top_sol = min(top_level_pareto_front, key=lambda x: x[0][0])
            logger.info(f"         Best Top-Level Solution: AggAcc={best_top_sol[0][0]:.4f}, AggComp={best_top_sol[0][1]:.4f}")
        
        if len(group1_pareto_front) > 0:
            best_group1_sol = min(group1_pareto_front, key=lambda x: x[0][0])
            logger.info(f"         Best Group1 Solution: Dice={best_group1_sol[0][0]:.4f}, Entropy={best_group1_sol[0][1]:.4f}")
        
        if len(group2_pareto_front) > 0:
            best_group2_sol = min(group2_pareto_front, key=lambda x: x[0][0])
            logger.info(f"         Best Group2 Solution: Complexity={best_group2_sol[0][0]:.4f}, Transformer={best_group2_sol[0][1]:.4f}")

        # Step the scheduler after optimizer has stepped
        scheduler.step()

        # write
        writer.add_scalar('Train/W_loss', weight_loss_avg, epoch)
        writer.add_scalar('Train/A_loss', arch_loss_avg, epoch)
        writer.add_scalar('Train/Dice', md, epoch)
        
        # Write complexity losses to TensorBoard
        if args.enable_complexity_loss:
            writer.add_scalar('Train/Complexity_loss', complexity_loss_avg, epoch)
            writer.add_scalar('Train/Arch_complexity_loss', arch_complexity_loss_avg, epoch)
            writer.add_scalar('Train/Transformer_loss', transformer_loss_avg, epoch)
#        writer.add_scalar('Train/time_each_epoch',epoch_time , epoch)

        # infer
        if (epoch + 1) % args.infer_epoch == 0:
            genotype = model.genotype()
            logger.info('genotype = %s', genotype)
            val_loss, (vmr, vms, vmp, vmf, vmjc, vmd, vmacc) = infer(args, model, val_queue, criterion)
            logger.info("ValLoss1:{:.3f} ValAcc1:{:.3f}  ValDice1:{:.3f} ValJc1:{:.3f}".format(val_loss, vmacc, vmd, vmjc))
            #writer.add_scalar('Val/loss1', val_loss, epoch)
            writer.add_scalar('Val/coffecient', vmjc, epoch)
            writer.add_scalar('Val/JC1', vmd, epoch)

            if args.gen_max_child:
                args.gen_max_child_flag = True
                val_loss, (vmr, vms, vmp, vmf, vmjc, vmd, vmacc) = infer(args, model, val_queue, criterion)
                logger.info("ValLoss2:{:.3f} ValAcc2:{:.3f}  ValDice2:{:.3f} ValJc2:{:.3f}".format(val_loss, vmacc, vmd, vmjc))
                writer.add_scalar('Val/loss', val_loss, epoch)
                args.gen_max_child_flag = False

            is_best = True if (vmjc >= max_value) else False
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
            if not is_best:
                torch.save(state, os.path.join(save_model_path, "checkpoint.pth.tar"))
            else:
                torch.save(state, os.path.join(save_model_path, "checkpoint.pth.tar"))
                torch.save(state, os.path.join(save_model_path, "model_best.pth.tar"))

    # one stage end, we should change the operations num (divided 2)
    weight_down = F.softmax(model.arch_parameters()[0], dim=-1).data.cpu().numpy()
    weight_up = F.softmax(model.arch_parameters()[1], dim=-1).data.cpu().numpy()
    weight_normal = F.softmax(model.arch_parameters()[2], dim=-1).data.cpu().numpy()
    weight_network = F.softmax(model.arch_parameters()[3], dim=-1).data.cpu().numpy()
    weight_transformer = F.softmax(model.arch_parameters()[4], dim=-1).data.cpu().numpy()
    
    logger.info("alphas_down: \n{}".format(weight_down))
    logger.info("alphas_up: \n{}".format(weight_up))
    logger.info("alphas_normal: \n{}".format(weight_normal))
    logger.info("alphas_network: \n{}".format(weight_network))
    logger.info("alphas_transformer_connections: \n{}".format(weight_transformer))
    logger.info("\n" + "="*80)
    logger.info("TRANSFORMER CONNECTION ANALYSIS")
    logger.info("="*80)
    logger.info("Transformer connections (ON probability):")
    for i, prob_on in enumerate(weight_transformer[:, 1]):
        status = "[ON]" if prob_on > 0.5 else "[OFF]"
        logger.info(f"  Connection {i}: {prob_on:.4f} {status}")
    logger.info(f"Total transformers activated: {sum(1 for p in weight_transformer[:, 1] if p > 0.5)}/{len(weight_transformer)}")
    logger.info("="*80 + "\n")

    genotype = model.genotype()
    logger.info('Genotype: {}'.format(genotype))
    logger.info('Transformer Connections: {}'.format(genotype.transformer_connections))

    # Log final HIERARCHICAL Pareto fronts
    logger.info("\n" + "="*80)
    logger.info("FINAL HIERARCHICAL PARETO FRONT ANALYSIS")
    logger.info("="*80)
    
    # === TOP-LEVEL PARETO (2D: Aggregated Accuracy vs Aggregated Complexity) ===
    logger.info(f"\n>>> TOP-LEVEL PARETO (2D): {len(top_level_pareto_front)} solutions")
    if len(top_level_pareto_front) > 0:
        logger.info("\nTop-Level Solutions (sorted by aggregated accuracy):")
        sorted_top = sorted(top_level_pareto_front, key=lambda x: x[0][0])
        
        for idx, (objectives, arch) in enumerate(sorted_top[:5]):  # Show top 5
            agg_acc, agg_comp = objectives
            logger.info(f"  Solution {idx+1}: AggAcc={agg_acc:.6f}, AggComp={agg_comp:.6f}")
        
        best_acc = min(top_level_pareto_front, key=lambda x: x[0][0])
        best_comp = min(top_level_pareto_front, key=lambda x: x[0][1])
        logger.info(f"\n  Best Accuracy:   AggAcc={best_acc[0][0]:.6f}, AggComp={best_acc[0][1]:.6f}")
        logger.info(f"  Best Simplicity: AggAcc={best_comp[0][0]:.6f}, AggComp={best_comp[0][1]:.6f}")
    
    # === GROUP 1 PARETO (Dice vs Entropy) ===
    logger.info(f"\n>>> GROUP 1 PARETO (Accuracy): {len(group1_pareto_front)} solutions")
    if len(group1_pareto_front) > 0:
        logger.info("Group 1 Solutions (Dice vs Entropy):")
        sorted_g1 = sorted(group1_pareto_front, key=lambda x: x[0][0])
        
        for idx, (objectives, arch) in enumerate(sorted_g1[:5]):  # Show top 5
            dice, entropy = objectives
            logger.info(f"  Solution {idx+1}: Dice={dice:.6f}, Entropy={entropy:.6f}")
        
        best_dice = min(group1_pareto_front, key=lambda x: x[0][0])
        best_entropy = min(group1_pareto_front, key=lambda x: x[0][1])
        logger.info(f"\n  Best Dice:    Dice={best_dice[0][0]:.6f}, Entropy={best_dice[0][1]:.6f}")
        logger.info(f"  Best Entropy: Dice={best_entropy[0][0]:.6f}, Entropy={best_entropy[0][1]:.6f}")
    
    # === GROUP 2 PARETO (Complexity vs Transformer) ===
    logger.info(f"\n>>> GROUP 2 PARETO (Complexity): {len(group2_pareto_front)} solutions")
    if len(group2_pareto_front) > 0:
        logger.info("Group 2 Solutions (Complexity vs Transformer):")
        sorted_g2 = sorted(group2_pareto_front, key=lambda x: x[0][0])
        
        for idx, (objectives, arch) in enumerate(sorted_g2[:5]):  # Show top 5
            comp, trans = objectives
            logger.info(f"  Solution {idx+1}: Complexity={comp:.6f}, Transformer={trans:.6f}")
        
        best_comp = min(group2_pareto_front, key=lambda x: x[0][0])
        best_trans = min(group2_pareto_front, key=lambda x: x[0][1])
        logger.info(f"\n  Best Complexity:   Complexity={best_comp[0][0]:.6f}, Transformer={best_comp[0][1]:.6f}")
        logger.info(f"  Best Transformer:  Complexity={best_trans[0][0]:.6f}, Transformer={best_trans[0][1]:.6f}")
    
    # Save all Pareto fronts to file
    pareto_save_path = os.path.join(args.save_path, 'hierarchical_pareto_fronts.pth')
    torch.save({
        'top_level_pareto_front': top_level_pareto_front,
        'group1_pareto_front': group1_pareto_front,
        'group2_pareto_front': group2_pareto_front,
        'final_genotype': genotype,
        'sorted_top_level': sorted(top_level_pareto_front, key=lambda x: x[0][0]) if top_level_pareto_front else []
    }, pareto_save_path)
    logger.info(f"\n\nHierarchical Pareto fronts saved to: {pareto_save_path}")
    logger.info("="*80 + "\n")

    writer.close()


def train(args, train_queue, val_queue, model, criterion, optimizer_weight, optimizer_arch, 
          group1_pareto_front, group2_pareto_front, top_level_pareto_front, ema_acc_loss, 
          train_arch, epoch=0, total_epochs=20):
    """
    Training function with HIERARCHICAL Pareto optimization.
    
    Args:
        group1_pareto_front: Pareto front for Group 1 (dice vs entropy)
        group2_pareto_front: Pareto front for Group 2 (complexity vs transformer)
        top_level_pareto_front: Top-level Pareto (aggregated_accuracy vs aggregated_complexity)
    """
    Train_recoder = BinaryIndicatorsMetric()
    w_loss_recoder = AverageMeter()
    a_loss_recoder = AverageMeter()
    complexity_loss_recoder = AverageMeter()
    arch_complexity_loss_recoder = AverageMeter()
    transformer_loss_recoder = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()

    for step, (input, target, _) in enumerate(train_queue):
        data_time.update(time.time() - end)

        input = input.to(args.device)
        target = target.to(args.device)

        # input is B C H W   target is B,1,H,W  preds: B,1,H,W
        optimizer_weight.zero_grad()
        preds = model(input,target,criterion)
        assert isinstance(preds, list)
        preds = [pred.view(pred.size(0), -1) for pred in preds]

        target = target.view(target.size(0), -1)

        torch.cuda.empty_cache()

        Train_recoder.update(labels=target, preds=preds[-1], n=1)

        if args.deepsupervision:
            for i in range(len(preds)):
                if i == 0:
                    target1_loss = criterion(preds[i], target)
                target1_loss += criterion(preds[i], target)
        else:
            target1_loss = criterion(preds[-1], target)

        w_loss = target1_loss
        
        # Note: Complexity loss is only for architecture optimization, not weight optimization
        # We only record it here for logging purposes, but don't include in weight loss
        if args.enable_complexity_loss and hasattr(model, 'alphas_normal'):
            with torch.no_grad():
                complexity_loss_for_logging = compute_complexity_loss(model, args)
                complexity_loss_recoder.update(complexity_loss_for_logging.item(), 1)

        # Backward pass for weights (only segmentation loss)
        w_loss.backward()

        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.weight_parameters(), args.grad_clip)

        optimizer_weight.step()

        w_loss_recoder.update(w_loss.item(), 1)

        # get all the indicators
        Train_recoder.update(labels=target, preds=preds[-1], n=1)

        # update network arch parameters
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above.
            try:
                input_search, target_search, _ = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(val_queue)
                input_search, target_search, _ = next(valid_queue_iter)
            input_search = input_search.to(args.device)
            target_search = target_search.to(args.device)

            optimizer_arch.zero_grad()
            archs_preds = model(input_search,target_search,criterion)
            archs_preds =[pred.view(pred.size(0), -1) for pred in archs_preds]
            target_search = target_search.view(target_search.size(0), -1)
            torch.cuda.empty_cache()
            
            # Compute accuracy loss ONCE (to be reused)
            if args.deepsupervision:
                accuracy_loss = sum(criterion(pred, target_search) for pred in archs_preds)
            else:
                accuracy_loss = criterion(archs_preds[-1], target_search)
            
            # Store accuracy loss value for Pareto tracking BEFORE using it in other computations
            accuracy_loss_value = accuracy_loss.item()
            
            # Compute complexity losses with progressive scheduling
            complexity_loss = compute_complexity_loss(model, args)
            arch_complexity_loss = compute_architecture_complexity_loss(model, args, epoch, total_epochs)
            
            # === COMPUTE DICE LOSS (needed for transformer optimization) ===
            if hasattr(model, 'alphas_transformer_connections'):
                pred_sigmoid = torch.sigmoid(archs_preds[-1])
                intersection = (pred_sigmoid * target_search).sum()
                union = pred_sigmoid.sum() + target_search.sum()
                dice = (2.0 * intersection + 1e-8) / (union + 1e-8)
                dice_loss_tensor = 1.0 - dice  # Dice loss with gradient
            
            # === GET PARETO TARGET FOR TRANSFORMER GUIDANCE ===
            # Extract best Dice solution from GROUP 1 Pareto front (Dice vs Entropy)
            pareto_target_dice = None
            if len(group1_pareto_front) > 0 and hasattr(model, 'alphas_transformer_connections'):
                # Find solution with best Dice (lowest dice_loss = first objective in Group 1)
                best_dice_solution = min(group1_pareto_front, key=lambda x: x[0][0])
                # Keep as SCALAR (float) - will be converted to tensor inside transformer loss function
                pareto_target_dice = best_dice_solution[0][0]
            
            # === COMPUTE TRANSFORMER LOSS WITH DIRECT GRADIENT & PARETO GUIDANCE ===
            if hasattr(model, 'alphas_transformer_connections'):
                transformer_loss, ema_acc_loss = compute_transformer_connection_loss(
                    model, args,
                    dice_loss_tensor=dice_loss_tensor,  # ← WITH GRADIENT!
                    ema_dice_loss=ema_acc_loss,
                    epoch=epoch,
                    total_epochs=total_epochs,
                    pareto_target_dice=pareto_target_dice  # ← PARETO GUIDANCE (scalar)
                )
            else:
                transformer_loss = torch.tensor(0.0, device=accuracy_loss.device)
            
            # === HIERARCHICAL PARETO MULTI-OBJECTIVE OPTIMIZATION ===
            # Get current architecture snapshot
            current_arch = get_architecture_snapshot(model)
            
            # Update HIERARCHICAL Pareto fronts (Level 1 + Level 2)
            group1_pareto_front, group2_pareto_front, top_level_pareto_front = hierarchical_pareto_optimization(
                dice_loss=accuracy_loss_value,  # Using accuracy_loss as dice metric
                entropy_loss=arch_complexity_loss.item(),
                complexity_loss=complexity_loss.item(),
                transformer_loss=transformer_loss.item(),
                group1_pareto_front=group1_pareto_front,
                group2_pareto_front=group2_pareto_front,
                top_level_pareto_front=top_level_pareto_front,
                architecture=current_arch
            )
            
            # Select target objectives from TOP-LEVEL Pareto front (2D: aggregated_accuracy vs aggregated_complexity)
            target_objectives = select_loss_from_pareto(top_level_pareto_front, args.pareto_strategy)
            
            # Use Pareto-guided weighting if we have a Pareto front
            if target_objectives is not None and args.enable_complexity_loss:
                # === HIERARCHICAL PARETO-GUIDED LOSS ===
                # target_objectives is 2D: (aggregated_accuracy, aggregated_complexity)
                a_loss = compute_pareto_guided_loss(
                    target_objectives, 
                    accuracy_loss,  # Dice loss tensor (with gradient)
                    complexity_loss,  # Complexity loss tensor (with gradient)
                    arch_complexity_loss,  # Entropy loss tensor (with gradient)
                    transformer_loss,  # Transformer loss tensor (with gradient)
                    args
                )
            else:
                # === FALLBACK: Hierarchical weighted sum ===
                a_loss = accuracy_loss
                if args.enable_complexity_loss:
                    # Use hierarchical aggregation even in fallback
                    group1_loss = 0.7 * accuracy_loss + 0.3 * arch_complexity_loss
                    group2_loss = 0.5 * complexity_loss + 0.5 * transformer_loss
                    a_loss = 0.5 * group1_loss + 0.5 * group2_loss
            
            # Record individual losses for logging
            a_loss_recoder.update(a_loss.item(), 1)
            complexity_loss_recoder.update(complexity_loss.item(), 1)
            arch_complexity_loss_recoder.update(arch_complexity_loss.item(), 1)
            transformer_loss_recoder.update(transformer_loss.item(), 1)

            # === CRITICAL: BACKWARD PASS FOR ARCHITECTURE PARAMETERS ===
            # This computes gradients for all architecture parameters including transformer connections
            a_loss.backward()
            
            if args.grad_clip:
                nn.utils.clip_grad_norm_(model.arch_parameters(), args.grad_clip)

            optimizer_arch.step()

    weight_loss_avg = w_loss_recoder.avg
    if train_arch:
        arch_loss_avg = a_loss_recoder.avg
    else:
        arch_loss_avg = 0
    
    # Get complexity loss averages
    complexity_loss_avg = complexity_loss_recoder.avg if complexity_loss_recoder.count > 0 else 0
    arch_complexity_loss_avg = arch_complexity_loss_recoder.avg if arch_complexity_loss_recoder.count > 0 else 0
    transformer_loss_avg = transformer_loss_recoder.avg if transformer_loss_recoder.count > 0 else 0
    
    mr, ms, mp, mf, mjc, md, macc = Train_recoder.get_avg
    batch_time.update(time.time() - end)
    end = time.time()
    print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time))

    return weight_loss_avg, arch_loss_avg, complexity_loss_avg, arch_complexity_loss_avg, transformer_loss_avg, mr, ms, mp, mf, mjc, md, macc, batch_time, group1_pareto_front, group2_pareto_front, top_level_pareto_front, ema_acc_loss


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
            preds = model(input,target,criterion)
            preds = [pred.view(pred.size(0), -1) for pred in preds]
            target = target.view(target.size(0), -1)
            if args.deepsupervision:
                for i in range(len(preds)):
                    if i == 0:
                        loss = criterion(preds[i], target)
                    loss += criterion(preds[i], target)
            else:
                loss = criterion(preds[-1], target)
            val_loss.update(loss.item(), 1)
            OtherVal.update(labels=target, preds=preds[-1], n=1)

            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time))
        return val_loss.avg, OtherVal.get_avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unet serieas Search')
    # Add default argument
    parser.add_argument('--model', type=str, default='Nas_Search_Unet',
                        help='Model to train and evaluation')
    parser.add_argument('--note', type=str, default='_', help="folder name note")
    parser.add_argument('--dataset', type=str, default='isic2018',
                        help='Model to train and evaluation')
    parser.add_argument('--dataset_root', type=str,
                        default='/mnt/data/KHTN2023/research25/hct-netm/datasets/cvc',
                        help='Model to train and evaluation')
    parser.add_argument('--base_size', type=int, default=256, help="resize base size")
    parser.add_argument('--crop_size', type=int, default=256, help="crop  size")
    parser.add_argument('--epochs', type=int, default=10, help="search epochs")
    parser.add_argument('--train_batch', type=int, default=2, help="train_batch")
    parser.add_argument('--val_batch', type=int, default=2, help="val_batch ")
    parser.add_argument('--num_workers', type=int, default=2, help="dataloader numworkers")
    parser.add_argument('--train_portion', type=float, default=0.5, help="dataloader numworkers")

    # search network setting
    parser.add_argument('--num_classes', type=int, default=1, help="output feature channel")
    parser.add_argument('--input_c', type=int, default=3, help="input img channel")
    parser.add_argument('--init_channel', type=int, default=16, help="init_channel for first leavel search cell")
    parser.add_argument('--meta_node_num', type=int, default=4, help="middle_nodes")
    parser.add_argument('--layers', type=int, default=9, help="layers")
    parser.add_argument('--use_sharing', type=bool, default=True,
                        help="The down op and up op have same normal operations")
    parser.add_argument('--depth', type=int, default=4, help="UnetFabrics`s layers and depth ")
    parser.add_argument('--double_down_channel', action='store_true', default=True, help="double_down_channel")
    parser.add_argument('--dropout_prob', type=float, default=0, help="dropout_prob")
    parser.add_argument('--use_softmax_head', type=bool, default=False, help='use_softmax_head')

    # model and device setting
    parser.add_argument('--init_weight_type', type=str, default="kaiming", help="the model init ")
    parser.add_argument('--arch_after', type=int, default=5,
                        help=" the first arch_after epochs without arch parameters traing")
    parser.add_argument('--infer_epoch', type=int, default=4, help=" val freq(epoch) ")
    parser.add_argument('--compute_freq', type=int, default=40, help=" compute freq(epoch) ")
    parser.add_argument('--gpus', type=int, default=1, help=" use cuda or not ")
    parser.add_argument('--grad_clip', type=int, default=1, help=" grid clip to ignore grad boom")
    parser.add_argument('--manualSeed', type=int, default=100, help=" manualSeed ")

    # seatch setting
    parser.add_argument('--loss', type=str, choices=['bcedice', 'bce', 'bcelog', 'dice', 'softdice', 'multibcedice'],
                        default="bcedice", help="loss name ")
    parser.add_argument('--dice_weight', type=int, default=10, help="dice loss weight in total loss")
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adm'], default='sgd',
                        help=" model_optimizer ! ")
    parser.add_argument('--deepsupervision', action='store_true', default=True, help=" deepsupervision nas uent ")
    # lr
    parser.add_argument('--lr', type=float, default=0.025, help="weight parameters lr ")
    parser.add_argument('--lr_min', type=float, default=1e-5, help=" min arch parameters lr  ")
    parser.add_argument('--weight_decay', type=float, default=3e-4, help=" for weight parameters lr  ")
    parser.add_argument('--arch_lr', type=float, default=1e-3, help="arch parameters lr ")
    parser.add_argument('--arch_weight_decay', type=float, default=0, help=" for arch parameters lr ")
    parser.add_argument('--momentum', type=float, default=0.9, help=" momentum  ")
    # resume
    parser.add_argument('--resume', type=str, default='/content/drive/MyDrive/MIxSearch12.31_dynamic_transformer/nas_search/search_exp/Nas_Search_Unet/isic2018/20220413-093911__/DSNAS/checkpoint.pth.tar', help=" resume file path")
    #DSNAS
    parser.add_argument('--early_fix_arch', action='store_true', default=True, help='bn affine flag')
    parser.add_argument('--gen_max_child', action='store_true', default=True,help='generate child network by argmax(alpha)')
    parser.add_argument('--gen_max_child_flag', action='store_true', default=False,help='flag of generating child network by argmax(alpha)')
    parser.add_argument('--random_sample', action='store_true', default=False, help='true if sample randomly')
    
    # Complexity loss weights
    parser.add_argument('--complexity_weight', type=float, default=0.2, help='weight for basic complexity loss')
    parser.add_argument('--arch_complexity_weight', type=float, default=0.1, help='weight for architecture entropy complexity loss')
    parser.add_argument('--transformer_connection_weight', type=float, default=1.0, help='weight for transformer connection REWARD (increased from 0.5)')
    parser.add_argument('--enable_complexity_loss', action='store_true', default=True, help='enable complexity loss during training')
    
    # Transformer warmup settings
    parser.add_argument('--transformer_warmup_epochs', type=int, default=5, help='number of epochs to force transformer exploration')
    parser.add_argument('--transformer_init_bias', type=float, default=0.3, help='initial bias for transformer ON state (0.3 = ~57%% initial ON probability, lower = more exploration)')
    parser.add_argument('--fix_transformer_arch', type=bool, default=False, help='whether to fix transformer connections early (default: False, let them learn)')
    parser.add_argument('--transformer_fix_margin', type=float, default=0.6, help='margin for fixing transformer decisions (higher = more confident, 0.6 vs 0.3 for operations)')
    parser.add_argument('--transformer_min_epochs_before_fix', type=int, default=15, help='minimum epochs before fixing transformers (warmup + learning time)')
    
    # Pareto multi-objective optimization
    parser.add_argument('--pareto_strategy', type=str, choices=['random', 'balanced', 'dice_focused'], 
                        default='dice_focused', help='strategy for selecting solution from Pareto front')
    
    args = parser.parse_args()
    main(args)
