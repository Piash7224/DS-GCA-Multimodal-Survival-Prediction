"""
DS-GCA FUSION WITH ATTENTION-BASED MIL (Mini-Batch & Grad Accumulation)


1. Loads .pt bag files directly from extracted folder
2. Mini-batching implemented to prevent CUDA OOM on Kaggle GPUs
3. Fixed validation indexing
4. CosineAnnealingLR instead of ReduceLROnPlateau
5. LayerNorm instead of BatchNorm in classifier
6. Dynamically loads fold-specific clinical encoders to prevent leakage
7. Gradient Accumulation applied for statistically stable Cox Loss computation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
import copy
import os
import random
import json


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


# CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LEARNING_RATE = 1e-4

#MINI-BATCH SIZE & GRADIENT ACCUMULATION
BATCH_SIZE = 16 
ACCUMULATION_STEPS = 4  # Effective batch size = 16 * 4 = 64

FUSION_DIM = 64
VIS_DIM = 512
CLIN_EMBED_DIM = 128  
PATIENCE = 20

# Paths
VIS_BAGS_DIR = "/kaggle/input/datasets/mahmudpiash/embeddings"
CLINICAL_CSV_PATH = "/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/clinical_data(labels).csv"
CLINICAL_ENCODER_DIR = "/kaggle/input/datasets/mahmudpiash/clinenc" 

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

print(f" Running DS-GCA + ABMIL on: {DEVICE} with BATCH_SIZE: {BATCH_SIZE} (Accumulated to {BATCH_SIZE * ACCUMULATION_STEPS})")


# LOAD BAGS FROM DIRECTORY

def load_bags_from_dir():
    print(f" Loading bags directly from {VIS_BAGS_DIR} ...")
    metadata_path = os.path.join(VIS_BAGS_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f" metadata.json not found in {VIS_BAGS_DIR}! Please check the path.")
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f" Metadata loaded: {len(metadata)} patients")

    bags = {}
    for pid in metadata.keys():
        pt_path = os.path.join(VIS_BAGS_DIR, f"{pid}.pt")
        if os.path.exists(pt_path):
            tensor = torch.load(pt_path, map_location='cpu').float()
            bags[pid] = tensor.numpy()

    print(f" Loaded {len(bags)} patient bags.")
    if len(bags) > 0:
        sizes = [bags[p].shape[0] for p in bags]
        print(f"   Bag sizes — min: {min(sizes)}, max: {max(sizes)}, mean: {np.mean(sizes):.1f}")
    return bags


# CLINICAL ENCODER

class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=CLIN_EMBED_DIM):
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

    def get_embedding(self, x):
        return self.encoder(x)


# ATTENTION-BASED MIL

class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        scores  = self.attention(x)
        weights = torch.softmax(scores, dim=0)
        aggregated = torch.sum(x * weights, dim=0)
        return aggregated


# GATED CROSS-ATTENTION

class GatedCrossAttention(nn.Module):
    def __init__(self, dim, heads=2, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.gate_linear = nn.Linear(dim * 2, dim)
        nn.init.constant_(self.gate_linear.bias, 0.5)
        self.gate    = nn.Sequential(self.gate_linear, nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key_value):
        attn_out, _ = self.attn(query=x_query, key=x_key_value, value=x_key_value)
        attn_out  = self.dropout(attn_out)
        gate_val  = self.gate(torch.cat([x_query, attn_out], dim=-1))
        out = gate_val * attn_out + (1 - gate_val) * x_query
        return self.norm(out)


# DS-GCA + MIL FUSION MODEL

class DualStreamFusionMIL(nn.Module):
    def __init__(self, clin_dim=CLIN_EMBED_DIM, vis_dim=VIS_DIM, fusion_dim=FUSION_DIM):
        super().__init__()
        self.visual_mil = AttentionMIL(feature_dim=vis_dim, hidden_dim=256)
        self.vis_proj = nn.Sequential(nn.Linear(vis_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.clin_proj = nn.Sequential(nn.Linear(clin_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.clin_query_vis = GatedCrossAttention(dim=fusion_dim, heads=2)
        self.vis_query_clin = GatedCrossAttention(dim=fusion_dim, heads=2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, vis_bag, clin):
        v_agg = self.visual_mil(vis_bag)
        v = self.vis_proj(v_agg).unsqueeze(0).unsqueeze(0)
        c = self.clin_proj(clin).unsqueeze(0).unsqueeze(0)

        c_enriched = self.clin_query_vis(c, v)
        v_enriched = self.vis_query_clin(v, c)

        fused = torch.cat([c_enriched.squeeze(0).squeeze(0), v_enriched.squeeze(0).squeeze(0)], dim=0)
        return self.classifier(fused.unsqueeze(0))


# COX LOSS

def cox_loss(risk_pred, time, event):
    risk_pred  = risk_pred.view(-1)
    order      = torch.argsort(time, descending=True)
    risk_sorted  = risk_pred[order]
    event_sorted = event[order]
    
    mask = event_sorted > 0
    if mask.sum() == 0:
        return risk_sorted.mean() * 0.0
        
    log_risk = torch.logcumsumexp(risk_sorted, dim=0)
    return -torch.sum((risk_sorted - log_risk) * event_sorted) / (event_sorted.sum() + 1e-8)


# CLINICAL DATA LOADING

def load_clinical_data():
    df = pd.read_csv(CLINICAL_CSV_PATH)
    df['clean_id'] = df['bcr_patient_barcode'].astype(str).str.slice(0, 12)
    df = df.set_index('clean_id').sort_index()

    def parse_event(x):
        return 1 if str(x) == '2' or str(x).lower() == 'dead' else 0

    y_event = df['vital_status'].apply(parse_event).values
    y_time  = df['Time'].values

    drop_cols = ['bcr_patient_barcode', 'Time', 'vital_status', 'time', 'event']
    df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    for c in df_feat.select_dtypes(['bool']):
        df_feat[c] = df_feat[c].astype(int)

    df_final = pd.get_dummies(df_feat, dummy_na=True).fillna(0)
    return df_final, y_time, y_event, df.index.tolist()


# ALIGN DATA

def load_and_align_data(bags):
    df_clinical, y_time, y_event, clinical_ids = load_clinical_data()
    vis_ids    = set(bags.keys())
    common_ids = sorted([
        pid for pid in vis_ids & set(clinical_ids)
        if pid not in EXCLUDE_PATIENTS
    ])

    aligned = []
    for pid in common_ids:
        cidx = clinical_ids.index(pid)
        aligned.append({
            'id':       pid,
            'clin_raw': df_clinical.iloc[cidx].values,
            'vis_bag':  bags[pid],
            'time':     y_time[cidx],
            'event':    y_event[cidx]
        })

    X_clin_raw = np.array([d['clin_raw'] for d in aligned])
    vis_bags   = [d['vis_bag']  for d in aligned]
    y_time_a   = np.array([d['time']     for d in aligned])
    y_event_a  = np.array([d['event']    for d in aligned])
    pids       = np.array([d['id']       for d in aligned])

    print(f" Aligned {len(aligned)} patients.")
    return X_clin_raw, vis_bags, y_time_a, y_event_a, pids


# TRAINING WITH 5-FOLD CV

def run_clean_cv_mil():
    bags = load_bags_from_dir()
    X_clin_raw, vis_bags, y_time, y_event, patient_ids = load_and_align_data(bags)
    clinical_input_dim = X_clin_raw.shape[1]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores      = []
    pooled_predictions = []
    best_global_ci   = 0.0
    best_global_model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_clin_raw, y_event)):
        print(f"\n FOLD {fold+1}/5")

        scaler = StandardScaler()
        X_clin_tr_sc = scaler.fit_transform(X_clin_raw[train_idx])
        X_clin_val_sc = scaler.transform(X_clin_raw[val_idx])

        clinical_encoder = ClinicalEncoder(input_dim=clinical_input_dim, embed_dim=CLIN_EMBED_DIM)
        
        encoder_filename = f"clinical_encoder_{CLIN_EMBED_DIM}d_fold{fold}.pth"
        encoder_path = os.path.join(CLINICAL_ENCODER_DIR, encoder_filename)
        
        print(f"  Loading clinical encoder: {encoder_filename}")
        if not os.path.exists(encoder_path):
             raise FileNotFoundError(f" Could not find {encoder_path}. Make sure CLINICAL_ENCODER_DIR is correct.")
             
        state_dict = torch.load(encoder_path, map_location=DEVICE)
        enc_dict   = {k: v for k, v in state_dict.items() if k.startswith("encoder")}
        clinical_encoder.load_state_dict(enc_dict, strict=False)
        clinical_encoder.to(DEVICE).eval()

        with torch.no_grad():
            train_clin_emb = clinical_encoder.get_embedding(
                torch.FloatTensor(X_clin_tr_sc).to(DEVICE)
            ).cpu().numpy()
            val_clin_emb = clinical_encoder.get_embedding(
                torch.FloatTensor(X_clin_val_sc).to(DEVICE)
            ).cpu().numpy()

        train_bags = [vis_bags[i] for i in train_idx]
        val_bags   = [vis_bags[i] for i in val_idx]

        model = DualStreamFusionMIL(clin_dim=CLIN_EMBED_DIM, vis_dim=VIS_DIM, fusion_dim=FUSION_DIM).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_val_ci   = 0.0
        best_state    = None
        patience_cnt  = 0

        for epoch in range(EPOCHS):
            model.train()
            perm = np.random.permutation(len(train_idx))

            #  MINI-BATCH WITH GRADIENT ACCUMULATION (Effective Batch = 64)
            optimizer.zero_grad() 
            accumulation_counter = 0

            for step in range(0, len(perm), BATCH_SIZE):
                batch_indices = perm[step : step + BATCH_SIZE]
                
                all_risks  = []
                all_times  = []
                all_events = []

                for i in batch_indices:
                    vis_bag = torch.FloatTensor(train_bags[i]).to(DEVICE)
                    clin    = torch.FloatTensor(train_clin_emb[i]).to(DEVICE)

                    risk = model(vis_bag, clin)
                    all_risks.append(risk)
                    all_times.append(y_time[train_idx[i]])
                    all_events.append(y_event[train_idx[i]])

                risks_t  = torch.cat(all_risks).view(-1)
                times_t  = torch.FloatTensor(all_times).to(DEVICE)
                events_t = torch.FloatTensor(all_events).to(DEVICE)

                # Calculate loss and scale it down by accumulation steps
                loss = cox_loss(risks_t, times_t, events_t)
                loss = loss / ACCUMULATION_STEPS
                
                # Accumulate gradients
                loss.backward()
                accumulation_counter += 1
                
                # Step optimizer ONLY when we hit the target OR it's the last batch
                is_last_batch = (step + BATCH_SIZE) >= len(perm)
                
                if accumulation_counter == ACCUMULATION_STEPS or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad() # Reset for the next round
                    accumulation_counter = 0
                
            scheduler.step()

            #  Validation (Full-Batch)
            model.eval()
            val_risks_list = []

            with torch.no_grad():
                for i in range(len(val_idx)):
                    vis_bag = torch.FloatTensor(val_bags[i]).to(DEVICE)
                    clin    = torch.FloatTensor(val_clin_emb[i]).to(DEVICE)
                    risk    = model(vis_bag, clin).item()
                    val_risks_list.append(risk)

            val_ci = concordance_index(
                y_time[val_idx],
                -np.array(val_risks_list),
                y_event[val_idx]
            )

            if val_ci > best_val_ci:
                best_val_ci = val_ci
                best_state  = copy.deepcopy(model.state_dict())
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        print(f"  Fold {fold+1} Best Val CI: {best_val_ci:.4f}")
        fold_scores.append(best_val_ci)
        torch.save(best_state, f"DSGCA_MIL_fold{fold+1}.pth")

        if best_val_ci > best_global_ci:
            best_global_ci    = best_val_ci
            best_global_model = copy.deepcopy(best_state)

        # Save pooled predictions
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            for i, idx in enumerate(val_idx):
                vis_bag = torch.FloatTensor(val_bags[i]).to(DEVICE)
                clin    = torch.FloatTensor(val_clin_emb[i]).to(DEVICE)
                risk    = model(vis_bag, clin).item()
                pooled_predictions.append({
                    'PatientID':    patient_ids[idx],
                    'RiskScore':    risk,
                    'SurvivalTime': y_time[idx],
                    'Event':        y_event[idx],
                    'Fold':         fold + 1
                })

    if best_global_model is not None:
        torch.save(best_global_model, "DSGCA_MIL_overall_best.pth")
        print(f"\n Best model saved. Val CI: {best_global_ci:.4f}")

    df_pooled = pd.DataFrame(pooled_predictions)
    df_pooled.to_csv("cv_results_mil_pooled.csv", index=False)

    avg = np.mean(fold_scores)
    std = np.std(fold_scores)
    print(f"\n FINAL DS-GCA + ABMIL CV: {avg:.4f} ± {std:.4f}")
    return fold_scores, df_pooled


if __name__ == "__main__":
    fold_scores, df_pooled = run_clean_cv_mil()