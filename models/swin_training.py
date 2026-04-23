#swin transformer training script for 4 classes. 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from sklearn.metrics import classification_report, f1_score
from PIL import Image
import numpy as np
import time
import os
import random
import torch.nn.functional as F


DATA_DIR = "/kaggle/working/dataset_patches"
BATCH_SIZE = 16
ACCUMULATE_STEPS = 4
LEARNING_RATE = 1e-4
EPOCHS = 40
NUM_CLASSES = 4
EMBED_DIM = 512
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
MIX_PROB = 0.5

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = "swin_s_512_4class_final.pth"
PATIENCE = 8

CLASS_NAMES = [
    "tumor",                   # 0
    "stroma",                  # 1
    "lymphocytic_infiltrate",  # 2
    "necrosis_or_debris",      # 3
]
# =================================================

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🌱 Random Seed set to: {seed}")

seed_everything(SEED)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f" Training Swin-Small (4-Class TME) on: {DEVICE}")


# MIXUP / CUTMIX

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mix_data(x, y, alpha=1.0, use_cutmix=False):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(DEVICE)
    if use_cutmix:
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2-bbx1)*(bby2-bby1) / (x.size()[-1]*x.size()[-2]))
    else:
        x = lam * x + (1 - lam) * x[index, :]
    return x, y, y[index], lam


#  FOCAL LOSS — no class weights (sampler handles balance)
# gamma focuses on hard examples only

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def focal_loss(self, inputs, targets):
        # No alpha/weight parameter — sampler already balances classes
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return torch.mean((1 - pt) ** self.gamma * ce_loss)

    def forward(self, inputs, targets_a, targets_b, lam):
        return (lam       * self.focal_loss(inputs, targets_a) +
                (1 - lam) * self.focal_loss(inputs, targets_b))


# DATASET

class BCSSPatchDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.targets = []

        print(f"\n  Loading from: {root}")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            folder_path = os.path.join(root, class_name)
            if not os.path.isdir(folder_path):
                print(f" Missing folder: {folder_path}")
                continue
            count = 0
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.samples.append(
                        (os.path.join(folder_path, fname), class_idx))
                    self.targets.append(class_idx)
                    count += 1
            print(f"  {'✅' if count > 0 else '❌'} "
                  f"{class_name:30s} [idx={class_idx}]: {count} patches")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ==========================================
# MODEL
# ==========================================
class Swin_4Class(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = models.swin_s(weights=models.Swin_S_Weights.DEFAULT)
        in_features = self.swin.head.in_features

        self.projection = nn.Linear(in_features, EMBED_DIM)
        self.activation = nn.ReLU()
        self.dropout    = nn.Dropout(0.5)
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)

        self.swin.head = nn.Sequential(
            self.projection,
            self.activation,
            self.dropout,
            self.classifier
        )

    def forward(self, x):
        return self.swin(x)

# ==========================================
# DATA LOADING
# ==========================================
def get_data():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("\n TRAIN SET:")
    train_data = BCSSPatchDataset(f"{DATA_DIR}/train", transform=train_transform)
    print("\n VAL SET:")
    val_data   = BCSSPatchDataset(f"{DATA_DIR}/val",   transform=val_transform)

    #WeightedRandomSampler — sole mechanism for class balancing
    targets      = np.array(train_data.targets)
    class_counts = np.bincount(targets, minlength=NUM_CLASSES)

    print(f"\n Train class counts:")
    for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
        print(f"   [{i}] {name:30s}: {count}")

    # Inverse frequency — sampler will draw each class equally per batch
    sample_weights_per_class = 1.0 / (class_counts + 1e-6)
    sample_weights = [sample_weights_per_class[t] for t in targets]
    sampler = WeightedRandomSampler(
        sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader

# ==========================================
# TRAINING LOOP
# ==========================================
def train_4class():
    train_loader, val_loader = get_data()

    model     = Swin_4Class().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                            optimizer, T_0=10, T_mult=2)

    # FocalLoss only — no alpha, no class weights
    criterion = FocalLoss(gamma=2.0)

    best_val_f1       = 0.0
    epochs_no_improve = 0

    print("\n🏁 Starting 4-Class TME Training...")
    print(f"   Classes : {CLASS_NAMES}")
    print(f"   Strategy: WeightedRandomSampler (balance) + FocalLoss gamma=2 (hard examples)\n")

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (img, label) in enumerate(train_loader):
            img, label = img.to(DEVICE), label.to(DEVICE)

            if np.random.rand() < MIX_PROB:
                use_cutmix = np.random.rand() < 0.5
                img, y_a, y_b, lam = mix_data(
                    img, label,
                    alpha=CUTMIX_ALPHA if use_cutmix else MIXUP_ALPHA,
                    use_cutmix=use_cutmix)
                loss = criterion(model(img), y_a, y_b, lam)
            else:
                loss = criterion(model(img), label, label, 1.0)

            (loss / ACCUMULATE_STEPS).backward()
            train_loss += loss.item()

            if (i + 1) % ACCUMULATE_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        if len(train_loader) % ACCUMULATE_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                _, pred = torch.max(model(img), 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        val_f1 = f1_score(all_labels, all_preds, average='macro',
                          labels=list(range(NUM_CLASSES)), zero_division=0)
        scheduler.step(epoch)

        duration = time.time() - start
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Macro-F1: {val_f1:.4f} | "
              f"Time: {duration:.0f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(" New Best Model Saved!")
            print(classification_report(
                all_labels, all_preds,
                target_names=CLASS_NAMES,
                labels=list(range(NUM_CLASSES)),
                digits=4, zero_division=0
            ))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f" Early stopping at epoch {epoch+1}")
                break

    print(f"\n Finished. Best Val Macro-F1: {best_val_f1:.4f}")
    print(f" Saved: {BEST_MODEL_PATH}")

if __name__ == "__main__":
    train_4class()