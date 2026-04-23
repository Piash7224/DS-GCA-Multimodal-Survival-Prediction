"""
FEATURE EXTRACTION: Memory-Optimized MIL Patch Bags (With Auto-Resume)

"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import shutil
import json
import zipfile


MODEL_PATH = "/kaggle/input/datasets/mohammadmahmudpiash/4class-swinmodel/swin_s_512_4class_final.pth"
DATA_DIR = "/kaggle/input/tcga-brca-survival-analysis/"

# UPLOADED CHECKPOINT PATH if resuming from a previous run
UPLOADED_CHECKPOINT_ZIP = "" 

OUTPUT_DIR = "/kaggle/working/patient_bags"
ZIP_PATH = "/kaggle/working/tcga_patient_features_512d_bags_FINAL"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WORKERS = 2
EMBED_DIM = 512
IMG_SIZE = 224
NUM_CLASSES = 4


# AUTO-RESUME SETUP (Smart Copy/Unzip)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if UPLOADED_CHECKPOINT_ZIP and os.path.exists(UPLOADED_CHECKPOINT_ZIP):
    # Check if working directory is already populated
    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')]
    
    if len(existing_files) == 0:
        if os.path.isfile(UPLOADED_CHECKPOINT_ZIP) and UPLOADED_CHECKPOINT_ZIP.endswith('.zip'):
            print(f" Found uploaded zip checkpoint! Unzipping to working directory...")
            with zipfile.ZipFile(UPLOADED_CHECKPOINT_ZIP, 'r') as zf:
                zf.extractall(OUTPUT_DIR)
            print(" Unzip complete. Ready to resume!")
            
        elif os.path.isdir(UPLOADED_CHECKPOINT_ZIP):
            print(f" Found uploaded folder checkpoint (Kaggle auto-unzipped)! Copying files...")
            for item in os.listdir(UPLOADED_CHECKPOINT_ZIP):
                s = os.path.join(UPLOADED_CHECKPOINT_ZIP, item)
                d = os.path.join(OUTPUT_DIR, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
            print(" Copy complete. Ready to resume!")
    else:
        print(" Files already exist in working directory. Skipping restore.")


# 1. MODEL DEFINITION

class Swin_Thesis_Max(nn.Module):
    def __init__(self):
        super(Swin_Thesis_Max, self).__init__()
        self.swin = models.swin_s(weights=None)
        in_features = self.swin.head.in_features
        
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

    def forward_features(self, x):
        """Extract 512-d features (before classifier)"""
        x = self.swin.features(x)
        x = self.swin.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.swin.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)  # 512-d features
        return x


# 2. DATASET DEFINITION

transform_pipeline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PatchDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.step = 250
        self.grid_size = 4

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            sub_patches = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    box = (i*self.step, j*self.step, (i+1)*self.step, (j+1)*self.step)
                    sub_patches.append(transform_pipeline(img.crop(box)))
            return torch.stack(sub_patches), True
        except Exception as e:
            return torch.zeros(16, 3, IMG_SIZE, IMG_SIZE), False


# 3. MAIN EXTRACTION

def main():
    print(f" Device: {DEVICE}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model not found at {MODEL_PATH}")
        return

    try:
        model = Swin_Thesis_Max()
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(" Model loaded successfully!")
    except Exception as e:
        print(f" Model load failed: {e}")
        return

    # Directory detection
    wsi_path = os.path.join(DATA_DIR, "WSIs")
    target_dir = wsi_path if os.path.exists(wsi_path) else DATA_DIR
    all_items = os.listdir(target_dir)
    patient_ids = [d for d in all_items if d.startswith("TCGA") and os.path.isdir(os.path.join(target_dir, d))]
    
    # Load Metadata & Resume Logic
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    processed_files = set(os.listdir(OUTPUT_DIR))
    processed_pids = set([f.replace(".pt", "") for f in processed_files if f.endswith(".pt")])
    
    print(f" Resuming... {len(processed_pids)} patients already done.")

    patients_to_process = [pid for pid in patient_ids if pid not in processed_pids]
    print(f" Remaining to process: {len(patients_to_process)}")

    if len(patients_to_process) == 0:
        print(" All patients processed!")
    else:
        # PROCESSING LOOP
        for i, pid in enumerate(tqdm(patients_to_process, desc="Processing")):
            patient_folder = os.path.join(target_dir, pid)
            
            # Gather images
            images = []
            dir_contents = os.listdir(patient_folder)
            images = [os.path.join(patient_folder, f) for f in dir_contents 
                      if f.lower().endswith(('.png', '.jpg', '.tif'))]
            
            if len(images) == 0:
                for root, _, files in os.walk(patient_folder):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.tif')):
                            images.append(os.path.join(root, f))
                    if len(images) > 0:
                        break
            
            if len(images) == 0:
                continue

            # Process patches
            dataset = PatchDataset(images)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=NUM_WORKERS, pin_memory=True)
            
            all_patch_features = []
            
            for batch_imgs, valid_flags in loader:
                if not valid_flags.any():
                    continue
                
                real_imgs = batch_imgs[valid_flags]
                B, N, C, H, W = real_imgs.shape
                real_imgs = real_imgs.view(-1, C, H, W).to(DEVICE)
                
                with torch.no_grad():
                    feats = model.forward_features(real_imgs)  # Shape: (B*N, 512)
                
                feats = feats.view(B, N, -1)
                
                for patch_bag in feats: 
                    all_patch_features.append(patch_bag.cpu().numpy())
            
            if len(all_patch_features) == 0:
                print(f" WARNING: Patient {pid} had {len(images)} images but ALL failed processing.")
                continue

            # Float16 Downcasting + Individual Saving
            all_patches = np.concatenate(all_patch_features, axis=0).astype(np.float16)
            bag_size = all_patches.shape[0]
            
            if np.all(all_patches == 0):
                continue
            
            # Save individual .pt
            tensor_bag = torch.tensor(all_patches, dtype=torch.float16)
            torch.save(tensor_bag, os.path.join(OUTPUT_DIR, f"{pid}.pt"))
            
            # Update metadata
            metadata[pid] = {'bag_size': bag_size}
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)

    
    # 4. FINAL ZIPPING (Runs only when ALL patients are done)
   
    processed_pids = [f.replace(".pt", "") for f in os.listdir(OUTPUT_DIR) if f.endswith(".pt")]
    print(f"\n Finished Extraction! Total processed patients: {len(processed_pids)}")
    
    if len(processed_pids) > 0 and len(patients_to_process) == 0: # Note: Wait for all processing to finish to zip
        print("\n Zipping folder to compress disk footprint...")
        shutil.make_archive(ZIP_PATH, 'zip', OUTPUT_DIR)
        print(f" Created {ZIP_PATH}.zip")
        
        print(" Cleaning up uncompressed files to free Kaggle space...")
        shutil.rmtree(OUTPUT_DIR)
        print(" ALL DONE! You are ready to train the fusion model.")

if __name__ == "__main__":
    main()