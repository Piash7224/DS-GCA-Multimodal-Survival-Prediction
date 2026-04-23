import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image


DATA_DIR = "/kaggle/working/dataset_patches"
MODEL_PATH = "/kaggle/working/swin_s_512_4class_final.pth" 
BATCH_SIZE = 16
NUM_CLASSES = 4
EMBED_DIM = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explicitly define classes in the EXACT order from training!
CLASS_NAMES = [
    "tumor",                   # 0
    "stroma",                  # 1
    "lymphocytic_infiltrate",  # 2
    "necrosis_or_debris",      # 3
]
# =================================================

print(f" Running Inference on Test Set using: {DEVICE}")

# 1. Re-Define Model Structure 
class Swin_For_FeatureExtraction(nn.Module):
    def __init__(self):
        super(Swin_For_FeatureExtraction, self).__init__()
        
        # Swin-S architecture
        self.swin = models.swin_s(weights=None) 
        
        in_features = self.swin.head.in_features
        
        # Re-create the custom head
        self.projection = nn.Linear(in_features, EMBED_DIM)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)
        
        self.swin.head = nn.Sequential(
            self.projection,
            self.activation,
            self.dropout,
            self.classifier
        )

    def forward(self, x):
        return self.swin(x)

# 2. Bring over custom Dataset to enforce label order
class BCSSPatchDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.targets = []

        print(f"\n Loading from: {root}")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            folder_path = os.path.join(root, class_name)
            if not os.path.isdir(folder_path):
                print(f"  Missing folder: {folder_path}")
                continue
            count = 0
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.samples.append((os.path.join(folder_path, fname), class_idx))
                    self.targets.append(class_idx)
                    count += 1
            print(f" {class_name:30s} [idx={class_idx}]: {count} patches")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# 3. Load Test Data Using Custom Dataset
def get_test_loader():
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_path = os.path.join(DATA_DIR, "test")
    if not os.path.exists(test_path):
        if os.path.exists(DATA_DIR):
             print(f"⚠️ 'test' subfolder not found in {DATA_DIR}. Checking root...")
        raise FileNotFoundError(f" Test folder not found at: {test_path}")
        
    #  Using the custom dataset 
    test_dataset = BCSSPatchDataset(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\n Total Test Samples: {len(test_dataset)}")
    return test_loader, CLASS_NAMES

# 4. Evaluation Function
def evaluate_model():
    # Load Data
    try:
        test_loader, class_names = get_test_loader()
    except FileNotFoundError as e:
        print(e)
        return

    # Load Model
    model = Swin_For_FeatureExtraction().to(DEVICE)
    
    if os.path.exists(MODEL_PATH):
        try:
            # Load weights
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"\n Model loaded successfully from {MODEL_PATH}")
        except RuntimeError as e:
            print(f" Error loading state dict: {e}")
            print("\n TIP: If you still get a size mismatch, ensure 'EMBED_DIM' matches training.")
            return
    else:
        print(f" Model file not found at {MODEL_PATH}")
        return

    model.eval()
    
    y_true = []
    y_pred = []
    
    print(" Predicting...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
            if (i + 1) % 10 == 0:
                print(f"   Processed batch {i + 1}/{len(test_loader)}")
            
    # --- REPORTING ---
    acc = accuracy_score(y_true, y_pred)
    print(f"\n Test Set Accuracy: {acc*100:.2f}%")
    print("-" * 50)
    print(" Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("-" * 50)
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()

if __name__ == "__main__":
    evaluate_model()