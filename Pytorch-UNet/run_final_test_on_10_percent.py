import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import KvasirSegDataset
from unet import UNet
from evaluate import evaluate
import wandb
import csv

# === Settings ===
batch_size = 2
model_path = 'checkpoints/checkpoint_epoch10.pth'
val_txt_path = 'val_images.txt'
image_root = 'data/imgs'
mask_root = 'data/masks'

# === Device ===
device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')

# === Load val image paths from val_images.txt ===
with open(val_txt_path, 'r') as f:
    val_image_paths = [line.strip() for line in f.readlines()]

# === Define transform (same as training) ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# === Subset wrapper for val_image_paths ===
class KvasirSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, selected_paths):
        self.base = base_dataset
        self.selected_indices = [i for i, path in enumerate(base_dataset.images) if path in selected_paths]

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        return self.base[self.selected_indices[idx]]

# === Load full dataset and wrap 10% subset ===
full_dataset = KvasirSegDataset(image_root, mask_root, transform=transform)
val_dataset = KvasirSubset(full_dataset, val_image_paths)

# === Dataloader ===
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Model ===
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# === wandb Init ===
wandb.init(project='U-Net', name='final-test-on-10-percent', config={"batch_size": batch_size})

# === Evaluate ===
avg_dice, per_image_dices = evaluate(model, val_loader, device, amp=False, log_per_image=True)
print(f"\n Final Dice Score on 10% Held-Out Validation Set: {avg_dice:.4f}")
wandb.log({'final-10-percent-eval-dice': avg_dice})
wandb.finish()

# === Save Avg Dice Score to CSV ===
with open('final_eval_10_percent.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['metric', 'value'])
    writer.writerow(['Dice Score (10%)', round(avg_dice, 4)])
print("📁 Saved average Dice score to final_eval_10_percent.csv")

# === Save Per-Image Dice Scores to CSV ===
with open('per_image_scores_10_percent.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image Index', 'Dice Score'])
    for i, score in enumerate(per_image_dices):
        writer.writerow([i, round(score, 4)])
print(" Saved per-image Dice scores to per_image_scores_10_percent.csv")
