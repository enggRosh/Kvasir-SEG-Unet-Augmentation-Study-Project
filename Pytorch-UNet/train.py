import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from dataset import KvasirSegDataset  # Dataset loader
from utils.dice_score import dice_loss
from torchvision import transforms

# Data path
image_dir = Path('./data/imgs/')
mask_dir = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = KvasirSegDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        transform=transform
    )

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    if device.type == 'mps':
        grad_scaler = torch.amp.GradScaler(enabled=False)
        amp = False
    else:
        grad_scaler = torch.amp.GradScaler(enabled=amp)

    criterion = nn.BCEWithLogitsLoss() if model.n_classes == 1 else nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32).contiguous()
                true_masks = batch['mask'].to(device=device, dtype=torch.float32).contiguous()

                with torch.autocast(device_type=device.type, enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(F.sigmoid(masks_pred), true_masks, multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(F.softmax(masks_pred, dim=1).float(),
                                          F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                          multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and global_step % division_step == 0:
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)

                    logging.info(f'Validation Dice score: {val_score}')
                    experiment.log({'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'step': global_step,
                                    'epoch': epoch})

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on Kvasir-SEG dataset')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-5, dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=1)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model.to(device=device)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('OOM error! Try smaller batch size or enable AMP.')
        torch.cuda.empty_cache()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            amp=args.amp
        )
