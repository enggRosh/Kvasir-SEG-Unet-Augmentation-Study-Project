import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Ensure input and target are contiguous
    input = input.contiguous()
    target = target.contiguous()
    
    # Ensure input and target have the same shape
    assert input.size() == target.size(), f"Shape mismatch: {input.size()} vs {target.size()}"

    # If input has 4 dimensions [B, C, H, W], flatten to [B, C*H*W] for batch-wise computation
    if reduce_batch_first and input.dim() == 4:
        input = input.reshape(input.size(0), -1)
        target = target.reshape(target.size(0), -1)
    elif input.dim() == 3:
        input = input.reshape(input.size(0), -1)
        target = target.reshape(target.size(0), -1)
    elif input.dim() == 2:
        input = input.view(1, -1)
        target = target.view(1, -1)

    inter = 2 * (input * target).sum(dim=1)
    sets_sum = input.sum(dim=1) + target.sum(dim=1)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
