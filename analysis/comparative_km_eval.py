import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import json

# ==========================================
# 🔒 SEED & EXCLUSIONS
# ==========================================
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

EXCLUDE_PATIENTS = set([
    'TCGA-A1-A0SK', 'TCGA-A1-A0SP', 'TCGA-A2-A04P', 'TCGA-A2-A04Q', 'TCGA-A2-A04T', 'TCGA-A2-A0CM', 'TCGA-A2-A0D0', 'TCGA-A2-A0ST', 'TCGA-A2-A0SX', 'TCGA-A2-A0T0', 'TCGA-A2-A0T2', 'TCGA-A2-A0YE', 'TCGA-A2-A0YM', 'TCGA-A2-A1G6', 'TCGA-A2-A3XS', 'TCGA-A2-A3XT', 'TCGA-A2-A3XU', 'TCGA-A2-A3XX', 'TCGA-A2-A3XY', 'TCGA-A7-A0CE', 'TCGA-A7-A0DA', 'TCGA-A7-A26F', 'TCGA-A7-A26G', 'TCGA-A7-A26I', 'TCGA-A7-A4SD', 'TCGA-A7-A4SE', 'TCGA-A7-A5ZV', 'TCGA-A7-A6VW', 'TCGA-A7-A6VY', 'TCGA-A8-A07C', 'TCGA-A8-A07O', 'TCGA-A8-A09X', 'TCGA-AC-A2BK', 'TCGA-AC-A2QH', 'TCGA-AC-A2QJ', 'TCGA-AC-A6IW', 'TCGA-AC-A7VC', 'TCGA-AN-A0AL', 'TCGA-AN-A0AR', 'TCGA-AN-A0AT', 'TCGA-AN-A0G0', 'TCGA-AN-A0XU', 'TCGA-AO-A03U', 'TCGA-AO-A0J2', 'TCGA-AO-A0J4', 'TCGA-AO-A0J6', 'TCGA-AO-A124', 'TCGA-AO-A128', 'TCGA-AO-A129', 'TCGA-AO-A12F', 'TCGA-AO-A1KR', 'TCGA-AQ-A04J', 'TCGA-AQ-A54N', 'TCGA-AR-A0TS', 'TCGA-AR-A0TU', 'TCGA-AR-A0U1', 'TCGA-AR-A0U4', 'TCGA-AR-A1AI', 'TCGA-AR-A1AQ', 'TCGA-AR-A1AR', 'TCGA-AR-A1AY', 'TCGA-AR-A256', 'TCGA-AR-A2LH', 'TCGA-AR-A2LR', 'TCGA-AR-A5QQ', 'TCGA-BH-A0AV', 'TCGA-BH-A0B3', 'TCGA-BH-A0B9', 'TCGA-BH-A0BG', 'TCGA-BH-A0BL', 'TCGA-BH-A0BW', 'TCGA-BH-A0E0', 'TCGA-BH-A0E6', 'TCGA-BH-A0RX', 'TCGA-BH-A0WA', 'TCGA-BH-A18G', 'TCGA-BH-A18V', 'TCGA-BH-A1EW', 'TCGA-BH-A1F6', 'TCGA-BH-A1FC', 'TCGA-BH-A42U', 'TCGA-C8-A12V', 'TCGA-C8-A131', 'TCGA-C8-A1HJ', 'TCGA-C8-A26X', 'TCGA-C8-A27B', 'TCGA-C8-A3M7', 'TCGA-D8-A13Z', 'TCGA-D8-A142', 'TCGA-D8-A143', 'TCGA-D8-A147', 'TCGA-D8-A1JF', 'TCGA-D8-A1JG', 'TCGA-D8-A1JL', 'TCGA-D8-A1XK', 'TCGA-D8-A1XQ', 'TCGA-D8-A27F', 'TCGA-D8-A27H', 'TCGA-D8-A27M', 'TCGA-E2-A14N', 'TCGA-E2-A14R', 'TCGA-E2-A14X', 'TCGA-E2-A150', 'TCGA-E2-A158', 'TCGA-E2-A159', 'TCGA-E2-A1AZ', 'TCGA-E2-A1B6', 'TCGA-E2-A1L7', 'TCGA-E2-A1LH', 'TCGA-E2-A1LI', 'TCGA-E2-A1LK', 'TCGA-E2-A1LL', 'TCGA-E2-A1LS', 'TCGA-E2-A573', 'TCGA-E2-A574', 'TCGA-E9-A5FL', 'TCGA-EW-A1OV', 'TCGA-EW-A1OW', 'TCGA-EW-A1P1', 'TCGA-EW-A1P4', 'TCGA-EW-A1P7', 'TCGA-EW-A1P8', 'TCGA-EW-A1PB', 'TCGA-EW-A1PH', 'TCGA-EW-A3U0', 'TCGA-EW-A6SB', 'TCGA-GM-A2DB', 'TCGA-GM-A2DD', 'TCGA-GM-A2DF', 'TCGA-GM-A2DH', 'TCGA-GM-A2DI', 'TCGA-GM-A3XL', 'TCGA-HN-A2NL', 'TCGA-LL-A441', 'TCGA-LL-A5YO', 'TCGA-LL-A73Y', 'TCGA-LL-A740', 'TCGA-OL-A5D6', 'TCGA-OL-A5D7', 'TCGA-OL-A66I', 'TCGA-OL-A66P', 'TCGA-OL-A6VO', 'TCGA-OL-A97C', 'TCGA-S3-AA10', 'TCGA-S3-AA15'
])

# ==========================================
# ⚙️ CONFIG
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 
ACCUMULATION_STEPS = 4  
FUSION_DIM = 64
VIS_DIM = 512
CLIN_EMBED_DIM = 128
PATIENCE = 20

VIS_BAGS_DIR = "/kaggle/input/datasets/mahmudpiash/embeddings"
CLINICAL_CSV_PATH = "/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/clinical_data(labels).csv"
CLINICAL_ENCODER_DIR = "/kaggle/input/datasets/mahmudpiash/clinenc"
DSGCA_CSV_PATH = "/kaggle/working/cv_results_mil_pooled.csv" # Your saved results

# ==========================================
# 🧠 ARCHITECTURE MODULES
# ==========================================
class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=CLIN_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(64, embed_dim), nn.BatchNorm1d(embed_dim), nn.GELU()
        )
    def get_embedding(self, x):
        return self.encoder(x)

class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(0)
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=0)
        return torch.sum(x * weights, dim=0)

class GatedCrossAttention(nn.Module):
    def __init__(self, dim, heads=2, dropout=0.2, use_gate=True, use_attention=True):
        super().__init__()
        self.use_gate = use_gate
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        if use_gate:
            self.gate_linear = nn.Linear(dim * 2, dim)
            nn.init.constant_(self.gate_linear.bias, 0.5)
            self.gate = nn.Sequential(self.gate_linear, nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key_value):
        if self.use_attention:
            attn_out, _ = self.attn(query=x_query, key=x_key_value, value=x_key_value)
            attn_out = self.dropout(attn_out)
        else:
            attn_out = x_key_value 
            
        if self.use_gate:
            gate_val = self.gate(torch.cat([x_query, attn_out], dim=-1))
            out = gate_val * attn_out + (1 - gate_val) * x_query
        else:
            out = x_query + attn_out 

        return self.norm(out)

class AblationFusionModel(nn.Module):
    def __init__(self, mode, clin_dim=CLIN_EMBED_DIM, vis_dim=VIS_DIM, fusion_dim=FUSION_DIM):
        super().__init__()
        self.mode = mode
        
        if mode != 'CLINICAL_ONLY':
            self.visual_mil = AttentionMIL(feature_dim=vis_dim)
            self.vis_proj = nn.Sequential(nn.Linear(vis_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())

        if mode != 'VISUAL_ONLY':
            self.clin_proj = nn.Sequential(nn.Linear(clin_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())

        if mode in ['FULL_DSGCA', 'GCA_NO_GATES', 'GATE_ONLY']:
            use_gate = (mode in ['FULL_DSGCA', 'GATE_ONLY'])
            use_attention = (mode in ['FULL_DSGCA', 'GCA_NO_GATES'])
            self.clin_query_vis = GatedCrossAttention(dim=fusion_dim, use_gate=use_gate, use_attention=use_attention)
            self.vis_query_clin = GatedCrossAttention(dim=fusion_dim, use_gate=use_gate, use_attention=use_attention)
            classifier_in = fusion_dim * 2
            
        elif mode in ['EARLY_FUSION']:
            classifier_in = fusion_dim * 2
        elif mode == 'LATE_FUSION':
            classifier_in = fusion_dim
        else: # Unimodal
            classifier_in = fusion_dim

        if mode == 'LATE_FUSION':
            self.clin_classifier = nn.Sequential(nn.Linear(fusion_dim, 16), nn.LayerNorm(16), nn.GELU(), nn.Dropout(0.5), nn.Linear(16, 1))
            self.vis_classifier = nn.Sequential(nn.Linear(fusion_dim, 16), nn.LayerNorm(16), nn.GELU(), nn.Dropout(0.5), nn.Linear(16, 1))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.5), nn.Linear(32, 1)
            )

    def forward(self, vis_bag, clin):
        if self.mode != 'CLINICAL_ONLY':
            v_agg = self.visual_mil(vis_bag)
            v = self.vis_proj(v_agg).unsqueeze(0).unsqueeze(0)
        
        if self.mode != 'VISUAL_ONLY':
            c = self.clin_proj(clin).unsqueeze(0).unsqueeze(0)
            
        if self.mode == 'LATE_FUSION':
            pred_c = self.clin_classifier(c.squeeze(0).squeeze(0).unsqueeze(0))
            pred_v = self.vis_classifier(v.squeeze(0).squeeze(0).unsqueeze(0))
            return (pred_c + pred_v) / 2.0

        if self.mode in ['FULL_DSGCA', 'GCA_NO_GATES', 'GATE_ONLY']:
            c_f = self.clin_query_vis(c, v).squeeze(0).squeeze(0)
            v_f = self.vis_query_clin(v, c).squeeze(0).squeeze(0)
            fused = torch.cat([c_f, v_f], dim=0)
        elif self.mode == 'EARLY_FUSION':
            fused = torch.cat([c.squeeze(0).squeeze(0), v.squeeze(0).squeeze(0)], dim=0)
        elif self.mode == 'CLINICAL_ONLY':
            fused = c.squeeze(0).squeeze(0)
        else: 
            fused = v.squeeze(0).squeeze(0)

        return self.classifier(fused.unsqueeze(0))

def cox_loss(risk_pred, time, event):
    risk_pred  = risk_pred.view(-1)
    order      = torch.argsort(time, descending=True)
    risk_sorted  = risk_pred[order]
    event_sorted = event[order]
    mask = event_sorted > 0
    if mask.sum() == 0: return risk_sorted.mean() * 0.0
    log_risk = torch.logcumsumexp(risk_sorted, dim=0)
    return -torch.sum((risk_sorted - log_risk) * event_sorted) / (event_sorted.sum() + 1e-8)

# ==========================================
# 📊 DATA LOADING
# ==========================================
def load_clinical_data():
    df = pd.read_csv(CLINICAL_CSV_PATH)
    df['clean_id'] = df['bcr_patient_barcode'].astype(str).str.slice(0, 12)
    df = df.set_index('clean_id').sort_index()

    def parse_event(x): return 1 if str(x) == '2' or str(x).lower() == 'dead' else 0

    y_event = df['vital_status'].apply(parse_event).values
    y_time  = df['Time'].values

    drop_cols = ['bcr_patient_barcode', 'Time', 'vital_status', 'time', 'event']
    df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    for c in df_feat.select_dtypes(['bool']): df_feat[c] = df_feat[c].astype(int)

    df_final = pd.get_dummies(df_feat, dummy_na=True).fillna(0)
    return df_final, y_time, y_event, df.index.tolist()

def load_bags_from_dir():
    print(f"📂 Loading bags directly from {VIS_BAGS_DIR} ...")
    metadata_path = os.path.join(VIS_BAGS_DIR, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    bags = {}
    for pid in metadata.keys():
        pt_path = os.path.join(VIS_BAGS_DIR, f"{pid}.pt")
        if os.path.exists(pt_path):
            bags[pid] = torch.load(pt_path, map_location='cpu').float().numpy()
    return bags

def load_and_align_data(bags):
    df_clinical, y_time, y_event, clinical_ids = load_clinical_data()
    vis_ids = set(bags.keys())
    
    common_ids = sorted([pid for pid in vis_ids & set(clinical_ids) if pid not in EXCLUDE_PATIENTS])
    time_dict = dict(zip(clinical_ids, y_time))
    event_dict = dict(zip(clinical_ids, y_event))

    aligned = []
    for pid in common_ids:
        clin_raw = df_clinical.loc[pid].values
        if clin_raw.ndim > 1: clin_raw = clin_raw[0]
            
        aligned.append({'id': pid, 'clin_raw': clin_raw, 'vis_bag': bags[pid], 'time': time_dict[pid], 'event': event_dict[pid]})

    X_clin_raw = np.array([d['clin_raw'] for d in aligned])
    vis_bags   = [d['vis_bag']  for d in aligned]
    y_time_a   = np.array([d['time']     for d in aligned])
    y_event_a  = np.array([d['event']    for d in aligned])
    return X_clin_raw, vis_bags, y_time_a, y_event_a

# ==========================================
# 🚂 OOF EXTRACTION LOOP
# ==========================================
def get_oof_predictions(mode, X_clin_raw, vis_bags, y_time, y_event):
    print(f"\n🚀 STARTING MODEL: {mode}")
    clinical_input_dim = X_clin_raw.shape[1]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_risks = np.zeros(len(y_time))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_clin_raw, y_event)):
        scaler = StandardScaler()
        X_clin_tr_sc = scaler.fit_transform(X_clin_raw[train_idx])
        X_clin_val_sc = scaler.transform(X_clin_raw[val_idx])

        clinical_encoder = ClinicalEncoder(input_dim=clinical_input_dim, embed_dim=CLIN_EMBED_DIM)
        encoder_path = os.path.join(CLINICAL_ENCODER_DIR, f"clinical_encoder_{CLIN_EMBED_DIM}d_fold{fold}.pth")
        
        if os.path.exists(encoder_path):
            state_dict = torch.load(encoder_path, map_location=DEVICE)
            clinical_encoder.load_state_dict({k: v for k, v in state_dict.items() if k.startswith("encoder")}, strict=False)
        else: raise FileNotFoundError(f"❌ Encoder missing: {encoder_path}")
        clinical_encoder.to(DEVICE).eval()

        with torch.no_grad():
            train_clin_emb = clinical_encoder.get_embedding(torch.FloatTensor(X_clin_tr_sc).to(DEVICE)).cpu().numpy()
            val_clin_emb = clinical_encoder.get_embedding(torch.FloatTensor(X_clin_val_sc).to(DEVICE)).cpu().numpy()

        model = AblationFusionModel(mode=mode, clin_dim=CLIN_EMBED_DIM, vis_dim=VIS_DIM, fusion_dim=FUSION_DIM).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_val_ci = 0.0
        patience_cnt = 0
        best_fold_risks = np.zeros(len(val_idx))

        for epoch in range(EPOCHS):
            model.train()
            perm = np.random.permutation(len(train_idx))
            optimizer.zero_grad()
            accumulation_counter = 0

            for step in range(0, len(perm), BATCH_SIZE):
                batch_indices = perm[step: step + BATCH_SIZE]
                all_risks, all_times, all_events = [], [], []

                for local_i in batch_indices:
                    global_i = train_idx[local_i]
                    risk = model(torch.FloatTensor(vis_bags[global_i]).to(DEVICE), torch.FloatTensor(train_clin_emb[local_i]).to(DEVICE))
                    all_risks.append(risk.view(1)) 
                    all_times.append(y_time[global_i])
                    all_events.append(y_event[global_i])

                loss = cox_loss(torch.cat(all_risks).view(-1), torch.FloatTensor(all_times).to(DEVICE), torch.FloatTensor(all_events).to(DEVICE)) / ACCUMULATION_STEPS
                loss.backward()
                accumulation_counter += 1

                if accumulation_counter == ACCUMULATION_STEPS or (step + BATCH_SIZE) >= len(perm):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad() 
                    accumulation_counter = 0
                
            scheduler.step()

            model.eval()
            val_risks_list = []
            with torch.no_grad():
                for i, global_i in enumerate(val_idx):
                    val_risks_list.append(model(torch.FloatTensor(vis_bags[global_i]).to(DEVICE), torch.FloatTensor(val_clin_emb[i]).to(DEVICE)).item())
            
            val_ci = concordance_index(y_time[val_idx], -np.array(val_risks_list), y_event[val_idx])

            if val_ci > best_val_ci:
                best_val_ci = val_ci
                best_fold_risks = np.array(val_risks_list)
                patience_cnt = 0
            else: patience_cnt += 1

            if patience_cnt >= PATIENCE: break

        print(f"   🏆 Fold {fold+1} Best Val CI: {best_val_ci:.4f}")
        fold_scores.append(best_val_ci)
        oof_risks[val_idx] = best_fold_risks

    print(f"✅ FINAL {mode} CV: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    return oof_risks

# ==========================================
# 📈 CLINICAL METRICS (HR & LOG-RANK)
# ==========================================
def get_clinical_metrics(time, event, risks):
    median_risk = np.median(risks)
    risk_groups = (risks > median_risk).astype(int)
    ix_high = (risk_groups == 1)
    ix_low = (risk_groups == 0)
    
    # Log-Rank
    p_val = logrank_test(time[ix_high], time[ix_low], 
                         event_observed_A=event[ix_high], event_observed_B=event[ix_low]).p_value
    
    # Hazard Ratio
    df = pd.DataFrame({'time': time, 'event': event, 'group': risk_groups})
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col='time', event_col='event')
        hr = cph.hazard_ratios_['group']
    except:
        hr = float('nan')
        
    return ix_high, ix_low, p_val, hr

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    bags = load_bags_from_dir()
    X_clin_raw, vis_bags, y_time, y_event = load_and_align_data(bags)
    
    # 1. RUN CLINICAL ONLY MODEL
    print("\n" + "#"*50 + "\nSTEP 1: GETTING CLINICAL_ONLY PREDICTIONS\n" + "#"*50)
    clin_risks = get_oof_predictions('CLINICAL_ONLY', X_clin_raw, vis_bags, y_time, y_event)
    
    # 2. LOAD DS-GCA PREDICTIONS FROM CSV
    print("\n" + "#"*50 + "\nSTEP 2: LOADING DS-GCA FROM CSV\n" + "#"*50)
    df_dsgca = pd.read_csv(DSGCA_CSV_PATH)
    # Using exact column names from your specific CSV format
    df_dsgca = df_dsgca.dropna(subset=["SurvivalTime", "Event", "RiskScore"])
    
    full_risks = df_dsgca["RiskScore"].values
    y_time_csv = df_dsgca["SurvivalTime"].values
    y_event_csv = df_dsgca["Event"].values

    # 3. CALCULATE METRICS
    print("\n" + "#"*50 + "\nSTEP 3: CLINICAL VALIDATION METRICS\n" + "#"*50)
    full_high, full_low, p_full, hr_full = get_clinical_metrics(y_time_csv, y_event_csv, full_risks)
    clin_high, clin_low, p_clin, hr_clin = get_clinical_metrics(y_time, y_event, clin_risks)

    print(f"🏆 DS-GCA Hazard Ratio:      {hr_full:.2f} (Log-Rank p = {p_full:.2e})")
    print(f"🏥 CLINICAL ONLY Hazard Ratio: {hr_clin:.2f} (Log-Rank p = {p_clin:.2e})")
    print("="*50 + "\n")

    # 4. PLOT SIDE-BY-SIDE KAPLAN MEIER
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    kmf = KaplanMeierFitter()

    # --- Plot 1: CLINICAL ONLY ---
    kmf.fit(y_time[clin_high], event_observed=y_event[clin_high], label='High Risk')
    kmf.plot_survival_function(ax=axes[0], color='#d62728', ci_alpha=0.2, linewidth=2)
    kmf.fit(y_time[clin_low], event_observed=y_event[clin_low], label='Low Risk')
    kmf.plot_survival_function(ax=axes[0], color='#1f77b4', ci_alpha=0.2, linewidth=2)

    axes[0].set_title(f"Clinical-Only Baseline\nHR: {hr_clin:.2f} | Log-Rank p: {p_clin:.2e}", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Time (Days)", fontsize=12)
    axes[0].set_ylabel("Survival Probability", fontsize=12)
    axes[0].grid(alpha=0.3)

    # --- Plot 2: DS-GCA ---
    kmf.fit(y_time_csv[full_high], event_observed=y_event_csv[full_high], label='High Risk')
    kmf.plot_survival_function(ax=axes[1], color='#d62728', ci_alpha=0.2, linewidth=2)
    kmf.fit(y_time_csv[full_low], event_observed=y_event_csv[full_low], label='Low Risk')
    kmf.plot_survival_function(ax=axes[1], color='#1f77b4', ci_alpha=0.2, linewidth=2)

    axes[1].set_title(f"DS-GCA (Multimodal)\nHR: {hr_full:.2f} | Log-Rank p: {p_full:.2e}", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Time (Days)", fontsize=12)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Breast Cancer Risk Stratification: Clinical Baseline vs. DS-GCA", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("Comparative_KM_Curves.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Finished. Saved 'Comparative_KM_Curves.png' to output directory.")