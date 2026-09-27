import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import KvasirSegDataset
from unet import UNet
from evaluate import evaluate
import wandb
import csv

# Settings
val_percent = 0.1
batch_size = 2
model_path = 'checkpoints/checkpoint_epoch10.pth'

# Device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset = KvasirSegDataset('data/imgs', 'data/masks', transform=transform)

# Split
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# Dataloader for the 90% training portion
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Model
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Initialize wandb
wandb.init(project='U-Net', name='90-percent-train-eval', config={"batch_size": batch_size})

# Evaluate
mean_dice, per_image_scores = evaluate(model, train_loader, device, amp=False, log_per_image=True)
print(f"\n Dice Score on 90% Training Data: {mean_dice:.4f}")

# Log to wandb
wandb.log({'train-set-eval-dice': mean_dice})

# Save per-image scores to CSV
csv_path = 'dice_scores_train.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image Index', 'Dice Score'])
    for i, score in enumerate(per_image_scores):
        writer.writerow([i, score])
print(f" Saved per-image Dice scores to {csv_path}")

wandb.finish()
