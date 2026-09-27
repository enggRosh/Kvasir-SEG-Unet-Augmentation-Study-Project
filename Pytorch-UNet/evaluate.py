import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, log_per_image=False):
    import wandb  # placed here to avoid circular imports if wandb is not always used

    net.eval()
    dice_scores = []

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation round', unit='batch', leave=False)):
            images, masks_true = batch['image'], batch['mask']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_true = masks_true.to(device=device, dtype=torch.long)

            masks_pred = net(images)

            for i in range(images.shape[0]):  # Loop through each image in batch
                pred = masks_pred[i].unsqueeze(0)
                true = masks_true[i].unsqueeze(0)

                if net.n_classes == 1:
                    pred = (F.sigmoid(pred) > 0.5).float()
                    score = dice_coeff(pred, true, reduce_batch_first=False).item()
                else:
                    true = F.one_hot(true, net.n_classes).permute(0, 3, 1, 2).float()
                    pred = F.one_hot(pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    score = multiclass_dice_coeff(pred[:, 1:], true[:, 1:], reduce_batch_first=False).item()

                dice_scores.append(score)

                if log_per_image:
                    wandb.log({'dice_score_per_image': score, 'global_image_index': len(dice_scores)-1})

    net.train()
    mean_dice = sum(dice_scores) / max(len(dice_scores), 1)
    return mean_dice, dice_scores
