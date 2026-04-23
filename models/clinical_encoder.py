#clinical embeddings with 5 fold cv (new)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
import copy
import random
import os


# 🔒 SEED EVERYTHING

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed}. Results are strictly reproducible.")

seed_everything(42)


# CONFIGURATION

CLINICAL_CSV_PATH = "/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/clinical_data(labels).csv"
VIS_BAGS_DIR      = "/kaggle/input/datasets/mahmudpiash/embeddings" 

EMBED_DIMS    = [32, 64, 128]
LEARNING_RATE = 0.0005
EPOCHS        = 100
PATIENCE      = 20
N_SPLITS      = 5  #  5-Fold CV

DEVICE = torch.device("cpu")
print(f"⚙️ Training Clinical Branch on: {DEVICE}")

# Ensuring identical exclusions as Stage 3 to prevent data leakage
EXCLUDE_PATIENTS = set([
    'TCGA-A1-A0SK', 'TCGA-A1-A0SP', 'TCGA-A2-A04P', 'TCGA-A2-A04Q',
    'TCGA-A2-A04T', 'TCGA-A2-A0CM', 'TCGA-A2-A0D0', 'TCGA-A2-A0ST',
    'TCGA-A2-A0SX', 'TCGA-A2-A0T0', 'TCGA-A2-A0T2', 'TCGA-A2-A0YE',
    'TCGA-A2-A0YM', 'TCGA-A2-A1G6', 'TCGA-A2-A3XS', 'TCGA-A2-A3XT',
    'TCGA-A2-A3XU', 'TCGA-A2-A3XX', 'TCGA-A2-A3XY', 'TCGA-A7-A0CE',
    'TCGA-A7-A0DA', 'TCGA-A7-A26F', 'TCGA-A7-A26G', 'TCGA-A7-A26I',
    'TCGA-A7-A4SD', 'TCGA-A7-A4SE', 'TCGA-A7-A5ZV', 'TCGA-A7-A6VW',
    'TCGA-A7-A6VY', 'TCGA-A8-A07C', 'TCGA-A8-A07O', 'TCGA-A8-A09X',
    'TCGA-AC-A2BK', 'TCGA-AC-A2QH', 'TCGA-AC-A2QJ', 'TCGA-AC-A6IW',
    'TCGA-AC-A7VC', 'TCGA-AN-A0AL', 'TCGA-AN-A0AR', 'TCGA-AN-A0AT',
    'TCGA-AN-A0G0', 'TCGA-AN-A0XU', 'TCGA-AO-A03U', 'TCGA-AO-A0J2',
    'TCGA-AO-A0J4', 'TCGA-AO-A0J6', 'TCGA-AO-A124', 'TCGA-AO-A128',
    'TCGA-AO-A129', 'TCGA-AO-A12F', 'TCGA-AO-A1KR', 'TCGA-AQ-A04J',
    'TCGA-AQ-A54N', 'TCGA-AR-A0TS', 'TCGA-AR-A0TU', 'TCGA-AR-A0U1',
    'TCGA-AR-A0U4', 'TCGA-AR-A1AI', 'TCGA-AR-A1AQ', 'TCGA-AR-A1AR',
    'TCGA-AR-A1AY', 'TCGA-AR-A256', 'TCGA-AR-A2LH', 'TCGA-AR-A2LR',
    'TCGA-AR-A5QQ', 'TCGA-BH-A0AV', 'TCGA-BH-A0B3', 'TCGA-BH-A0B9',
    'TCGA-BH-A0BG', 'TCGA-BH-A0BL', 'TCGA-BH-A0BW', 'TCGA-BH-A0E0',
    'TCGA-BH-A0E6', 'TCGA-BH-A0RX', 'TCGA-BH-A0WA', 'TCGA-BH-A18G',
    'TCGA-BH-A18V', 'TCGA-BH-A1EW', 'TCGA-BH-A1F6', 'TCGA-BH-A1FC',
    'TCGA-BH-A42U', 'TCGA-C8-A12V', 'TCGA-C8-A131', 'TCGA-C8-A1HJ',
    'TCGA-C8-A26X', 'TCGA-C8-A27B', 'TCGA-C8-A3M7', 'TCGA-D8-A13Z',
    'TCGA-D8-A142', 'TCGA-D8-A143', 'TCGA-D8-A147', 'TCGA-D8-A1JF',
    'TCGA-D8-A1JG', 'TCGA-D8-A1JL', 'TCGA-D8-A1XK', 'TCGA-D8-A1XQ',
    'TCGA-D8-A27F', 'TCGA-D8-A27H', 'TCGA-D8-A27M', 'TCGA-E2-A14N',
    'TCGA-E2-A14R', 'TCGA-E2-A14X', 'TCGA-E2-A150', 'TCGA-E2-A158',
    'TCGA-E2-A159', 'TCGA-E2-A1AZ', 'TCGA-E2-A1B6', 'TCGA-E2-A1L7',
    'TCGA-E2-A1LH', 'TCGA-E2-A1LI', 'TCGA-E2-A1LK', 'TCGA-E2-A1LL',
    'TCGA-E2-A1LS', 'TCGA-E2-A573', 'TCGA-E2-A574', 'TCGA-E9-A5FL',
    'TCGA-EW-A1OV', 'TCGA-EW-A1OW', 'TCGA-EW-A1P1', 'TCGA-EW-A1P4',
    'TCGA-EW-A1P7', 'TCGA-EW-A1P8', 'TCGA-EW-A1PB', 'TCGA-EW-A1PH',
    'TCGA-EW-A3U0', 'TCGA-EW-A6SB', 'TCGA-GM-A2DB', 'TCGA-GM-A2DD',
    'TCGA-GM-A2DF', 'TCGA-GM-A2DH', 'TCGA-GM-A2DI', 'TCGA-GM-A3XL',
    'TCGA-HN-A2NL', 'TCGA-LL-A441', 'TCGA-LL-A5YO', 'TCGA-LL-A73Y',
    'TCGA-LL-A740', 'TCGA-OL-A5D6', 'TCGA-OL-A5D7', 'TCGA-OL-A66I',
    'TCGA-OL-A66P', 'TCGA-OL-A6VO', 'TCGA-OL-A97C', 'TCGA-S3-AA10',
    'TCGA-S3-AA15'
])


# MODEL

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        risk      = self.head(embedding)
        return risk, embedding


# FULL-BATCH COX LOSS

def cox_loss(risk_pred, y_time, y_event):
    risk_pred = risk_pred.view(-1)
    idx       = torch.argsort(y_time, descending=True)
    risk_pred = risk_pred[idx]
    y_event   = y_event[idx]

    mask = y_event > 0
    if mask.sum() == 0:
        return risk_pred.mean() * 0.0

    log_cum_sum = torch.logcumsumexp(risk_pred, dim=0)
    return -torch.sum((risk_pred - log_cum_sum) * y_event) / mask.sum()


# LOAD + PREPROCESS

def load_clinical_dataframe():
    print(" Loading Clinical CSV...")
    df = pd.read_csv(CLINICAL_CSV_PATH)
    df['clean_id'] = df['bcr_patient_barcode'].astype(str).str.slice(0, 12)
    
    #  Crucial: Sorting index ensures consistent fold splits across all scripts
    df = df.set_index('clean_id').sort_index()

    # : Filter out excluded patients BEFORE making splits
    initial_count = len(df)
    df = df[~df.index.isin(EXCLUDE_PATIENTS)]
    print(f"   Excluded {initial_count - len(df)} patients found in EXCLUDE_PATIENTS list.")

    # Filter out patients missing Image Bags
    if os.path.exists(VIS_BAGS_DIR):
        available_bags = set([f.replace(".pt", "") for f in os.listdir(VIS_BAGS_DIR) if f.endswith(".pt")])
        pre_bag_count = len(df)
        df = df[df.index.isin(available_bags)]
        print(f"   Excluded {pre_bag_count - len(df)} patients who are missing WSI .pt bags.")
    else:
        print(f"  WARNING: VIS_BAGS_DIR not found at {VIS_BAGS_DIR}. Cannot filter by image availability.")

    def parse_event(x):
        try:
            return 1 if str(x) == '2' or str(x).lower() == 'dead' else 0
        except:
            return 0

    y_event = df['vital_status'].apply(parse_event).values
    y_time  = df['Time'].values

    drop_cols = ['bcr_patient_barcode', 'Time', 'vital_status', 'time', 'event']
    df_feat = df.drop(
        columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    for c in df_feat.select_dtypes(['bool']):
        df_feat[c] = df_feat[c].astype(int)

    df_final = pd.get_dummies(df_feat, dummy_na=True).fillna(0)
    return df_final, y_time, y_event, df.index.tolist()


# 5-FOLD CV TRAINING

def run_comparison():
    df_features, y_time, y_event, pids = load_clinical_dataframe()
    X = df_features.values

    print(f"\n Dataset Info (Cleaned & Aligned for Fusion):")
    print(f"   Total patients  : {len(pids)}")
    print(f"   Feature dim     : {X.shape[1]}")
    print(f"   Events (deaths) : {y_event.sum()}")
    print(f"   Censored        : {len(y_event) - y_event.sum()}")

    #  Use StratifiedKFold to maintain death/censored ratios in each fold
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    results = {}

    print(f"\n 5-FOLD CV EMBEDDING DIMENSION COMPARISON: {EMBED_DIMS}")
    print("=" * 60)

    for dim in EMBED_DIMS:
        print(f"\n{'='*60}")
        print(f" Testing Embedding Dimension: {dim}")
        print(f"{'='*60}")
        
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_event)):
            print(f"\n  FOLD {fold + 1}/{N_SPLITS}")
            
            # Extract fold data 
            X_tr, X_val = X[train_idx], X[val_idx]
            y_t_tr, y_t_val = y_time[train_idx], y_time[val_idx]
            y_e_tr, y_e_val = y_event[train_idx], y_event[val_idx]

            # Scaler fit on train ONLY per fold 
            scaler = StandardScaler()
            X_tr  = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

            # Convert to full-batch tensors ─
            Xtr_t   = torch.FloatTensor(X_tr).to(DEVICE)
            ytt_t   = torch.FloatTensor(y_t_tr).to(DEVICE)
            yet_t   = torch.FloatTensor(y_e_tr).to(DEVICE)

            Xval_t  = torch.FloatTensor(X_val).to(DEVICE)

            #  Model & Optimizer Reset per fold 
            seed_everything(42 + fold) # slight offset per fold ensures identical init if run again
            model     = ClinicalEncoder(input_dim=X.shape[1], embed_dim=dim).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

            best_c           = 0.0
            best_model_wts   = copy.deepcopy(model.state_dict())
            patience_counter = 0

            for epoch in range(EPOCHS):
                # Train 
                model.train()
                optimizer.zero_grad()
                risk, _ = model(Xtr_t)
                loss = cox_loss(risk, ytt_t, yet_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                #  Validation
                model.eval()
                with torch.no_grad():
                    val_risk, _ = model(Xval_t)
                    val_risk_np = val_risk.cpu().numpy().flatten()

                try:
                    c = concordance_index(y_t_val, -val_risk_np, y_e_val)
                except:
                    c = 0.5

                if c > best_c:
                    best_c           = c
                    best_model_wts   = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    break

            print(f"  Best Validation CI: {best_c:.4f}")
            fold_scores.append(best_c)

            #  Save fold-specific weights
            model.load_state_dict(best_model_wts)
            model_filename = f"clinical_encoder_{dim}d_fold{fold}.pth"
            torch.save(model.state_dict(), model_filename)
        
        # Calculate mean CI for this embedding dimension
        mean_ci = np.mean(fold_scores)
        std_ci = np.std(fold_scores)
        print(f"\n  {dim}d Summary: Mean CI = {mean_ci:.4f} ± {std_ci:.4f}")
        results[dim] = mean_ci

    #  Final report 
    print("\n" + "=" * 60)
    print(" FINAL SCOREBOARD: 5-FOLD UNIMODAL PERFORMANCE")
    print("=" * 60)

    for dim, score in results.items():
        print(f"   Embedding Dimension {dim:3d}: Mean CI = {score:.4f}")

    winner = max(results, key=results.get)
    print("\n" + "-" * 60)
    print(f" BEST OVERALL: {winner}-Dimension  |  Mean CI: {results[winner]:.4f}")
    print("-" * 60)

   

    return results

if __name__ == "__main__":
    results = run_comparison()