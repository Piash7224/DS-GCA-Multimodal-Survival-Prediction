"""
DS-GCA COMPREHENSIVE FUSION AUDIT (Feature-Wise & Patient-Wise)
=============================================================
This script audits the trained FULL_DSGCA model to quantify 
the micro-adaptive behavior of the Gated Cross-Attention mechanism.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import spearmanr

# ==========================================
# ⚙️ CONFIGURATION — YOUR EXACT PATHS
# ==========================================
SOTA_MODEL_PATH       = "/kaggle/input/datasets/mahmudpiash/fusion/DSGCA_MIL_overall_best.pth" 
CLINICAL_ENCODER_PATH = "/kaggle/input/datasets/mahmudpiash/clinenc/clinical_encoder_128d_fold4.pth" 
VIS_BAGS_DIR          = "/kaggle/input/datasets/mahmudpiash/embeddings"
CLINICAL_CSV_PATH     = "/kaggle/input/datasets/jmalagontorres/tcga-brca-survival-analysis/clinical_data(labels).csv"

CLIN_EMBED_DIM = 128
VIS_DIM        = 512
FUSION_DIM     = 64

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

# ==========================================
# 1. ARCHITECTURE MODULES
# ==========================================
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

class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
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
            gate_val = torch.ones_like(x_query) 
            out = x_query + attn_out 

        return self.norm(out), gate_val

class AblationFusionModel(nn.Module):
    def __init__(self, mode='FULL_DSGCA', clin_dim=CLIN_EMBED_DIM, vis_dim=VIS_DIM, fusion_dim=FUSION_DIM):
        super().__init__()
        self.mode = mode
        self.visual_mil = AttentionMIL(feature_dim=vis_dim)
        self.vis_proj = nn.Sequential(nn.Linear(vis_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.clin_proj = nn.Sequential(nn.Linear(clin_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        
        self.clin_query_vis = GatedCrossAttention(dim=fusion_dim, heads=2, use_gate=True, use_attention=True)
        self.vis_query_clin = GatedCrossAttention(dim=fusion_dim, heads=2, use_gate=True, use_attention=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward_audit(self, vis_bag, clin):
        v_agg = self.visual_mil(vis_bag).squeeze()
        clin = clin.squeeze() 
        v = self.vis_proj(v_agg).unsqueeze(0).unsqueeze(0) 
        c = self.clin_proj(clin).unsqueeze(0).unsqueeze(0) 

        c_enriched, gate_cv = self.clin_query_vis(c, v)
        v_enriched, gate_vc = self.vis_query_clin(v, c)

        fused = torch.cat([c_enriched.squeeze(0).squeeze(0), v_enriched.squeeze(0).squeeze(0)], dim=0)
        risk = self.classifier(fused.unsqueeze(0))
        return gate_cv, gate_vc, risk

# ==========================================
# 2. DATA LOADERS
# ==========================================
def load_bags_from_dir():
    print(f"📂 Loading bags directly from {VIS_BAGS_DIR} ...")
    metadata_path = os.path.join(VIS_BAGS_DIR, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    bags = {}
    for pid in metadata.keys():
        pt_path = os.path.join(VIS_BAGS_DIR, f"{pid}.pt")
        if os.path.exists(pt_path):
            tensor = torch.load(pt_path, map_location='cpu').float()
            bags[pid] = tensor.numpy()
    return bags

def load_clinical_data():
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(CLINICAL_CSV_PATH)
    df['clean_id'] = df['bcr_patient_barcode'].astype(str).str.slice(0, 12)
    df = df.set_index('clean_id').sort_index()
    drop_cols = ['bcr_patient_barcode', 'Time', 'vital_status', 'time', 'event']
    df_feat = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    for c in df_feat.select_dtypes(['bool']):
        df_feat[c] = df_feat[c].astype(int)
    df_final = pd.get_dummies(df_feat, dummy_na=True).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_final.values)
    return X_scaled, df_final.index.tolist(), X_scaled.shape[1]

# ==========================================
# 3. ADVANCED STATISTICAL AUDIT
# ==========================================
def perform_statistical_audit(name, gate_tensor):
    """Calculates granular adaptivity metrics."""
    g = gate_tensor.squeeze(1).numpy() # Shape (N_Patients, D_Features)
    
    # Global Metrics
    mean_val = np.mean(g)
    std_val = np.std(g)
    
    # Adaptivity Metrics
    # How much do different patients weight the SAME feature differently?
    patient_adaptivity = np.mean(np.std(g, axis=0)) 
    # How much does one patient weight different features differently?
    feature_discrimination = np.mean(np.std(g, axis=1))
    
    # Saturation (Near 0 or 1)
    sat_pct = (np.sum(g > 0.95) + np.sum(g < 0.05)) / g.size * 100
    
    print(f"\nAUDIT: {name}")
    print(f"--------------------------------------------------")
    print(f"Global Mean        : {mean_val:.4f} (Expected: ~0.62)")
    print(f"Global Std Dev     : {std_val:.4f}")
    print(f"Range              : [{np.min(g):.4f} to {np.max(g):.4f}]")
    print(f"Patient Adaptivity : {patient_adaptivity:.4f} (Variation across people)")
    print(f"Feature Discrim.   : {feature_discrimination:.4f} (Variation across dims)")
    print(f"Saturation Level   : {sat_pct:.2f}%")
    
    return g

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def audit_mil_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 STARTING COMPREHENSIVE DSGCA AUDIT ON {device}")

    bags = load_bags_from_dir()
    X_clin_raw, clinical_ids, input_dim = load_clinical_data()
    
    vis_ids = set(bags.keys())
    common_ids = sorted([pid for pid in vis_ids & set(clinical_ids) if pid not in EXCLUDE_PATIENTS])
    print(f"✅ Aligned {len(common_ids)} patients.")

    clin_enc = ClinicalEncoder(input_dim=input_dim, embed_dim=CLIN_EMBED_DIM).to(device)
    state_dict = torch.load(CLINICAL_ENCODER_PATH, map_location=device)
    enc_dict = {k: v for k, v in state_dict.items() if k.startswith("encoder")}
    clin_enc.load_state_dict(enc_dict, strict=False)
    clin_enc.eval()

    model = AblationFusionModel(mode='FULL_DSGCA').to(device)
    model.load_state_dict(torch.load(SOTA_MODEL_PATH, map_location=device))
    model.eval()

    all_gate_cv, all_gate_vc, all_risks = [], [], []

    print(f"🔄 Auditing {len(common_ids)} patients...")
    with torch.no_grad():
        for pid in common_ids:
            cidx = clinical_ids.index(pid)
            clin_tensor = torch.FloatTensor(X_clin_raw[cidx]).unsqueeze(0).to(device)
            clin_emb = clin_enc.get_embedding(clin_tensor)
            vis_bag = torch.FloatTensor(bags[pid]).to(device)
            
            g_cv, g_vc, risk = model.forward_audit(vis_bag, clin_emb)
            
            all_gate_cv.append(g_cv.cpu())
            all_gate_vc.append(g_vc.cpu())
            all_risks.append(risk.cpu().item())

    # Convert to arrays
    g_cv = perform_statistical_audit("Stream 1 ($g_c$): Clinical queries Visual", torch.cat(all_gate_cv, 0))
    g_vc = perform_statistical_audit("Stream 2 ($g_v$): Visual queries Clinical", torch.cat(all_gate_vc, 0))
    risks = np.array(all_risks)

    # Risk-Gate Correlation Analysis
    # Does the gate mean change for high-risk patients?
    gate_means_per_pt = g_cv.mean(axis=1)
    corr, p_val = spearmanr(gate_means_per_pt, risks)
    print(f"\nCORRELATION ANALYSIS:")
    print(f"Spearman Correlation (Gate Mean vs Predicted Risk): {corr:.4f} (p={p_val:.4f})")

    # ==========================================
    # 🌟 PUBLICATION-READY 2x2 PLOT
    # ==========================================
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)
    fig.subplots_adjust(hspace=0.35, wspace=0.22)
    
    init_val = 1 / (1 + np.exp(-0.5))  # Sigmoid(0.5)

    # --- TOP ROW: GC ---
    # Feature-wise (A)
    gc_feat_mean = g_cv.mean(axis=0)
    gc_feat_std  = g_cv.std(axis=0)
    axes[0, 0].bar(np.arange(FUSION_DIM), gc_feat_mean, yerr=gc_feat_std, capsize=2, color='steelblue', alpha=0.8)
    axes[0, 0].axhline(init_val, color='crimson', linestyle='--', label=f'Init: {init_val:.3f}')
    axes[0, 0].set_ylim(0.45, 0.75) # ZOOMED to show adaptivity
    axes[0, 0].set_title("A: Feature-Wise Adaptivity ($g_c$)", fontweight='bold')
    axes[0, 0].set_ylabel("Mean Activation")

    # Patient-wise (B)
    sns.histplot(g_cv.mean(axis=1), kde=True, ax=axes[0, 1], color='steelblue')
    axes[0, 1].set_title("B: Patient-Wise Adaptivity ($g_c$)", fontweight='bold')
    axes[0, 1].set_xlabel("Average Gate per Patient")

    # --- BOTTOM ROW: GV ---
    # Feature-wise (C)
    gv_feat_mean = g_vc.mean(axis=0)
    gv_feat_std  = g_vc.std(axis=0)
    axes[1, 0].bar(np.arange(FUSION_DIM), gv_feat_mean, yerr=gv_feat_std, capsize=2, color='darkorange', alpha=0.8)
    axes[1, 0].axhline(init_val, color='crimson', linestyle='--', label=f'Init: {init_val:.3f}')
    axes[1, 0].set_ylim(0.45, 0.75) # ZOOMED
    axes[1, 0].set_title("C: Feature-Wise Adaptivity ($g_v$)", fontweight='bold')
    axes[1, 0].set_ylabel("Mean Activation")
    axes[1, 0].set_xlabel("Fusion Dimension")

    # Patient-wise (D)
    sns.histplot(g_vc.mean(axis=1), kde=True, ax=axes[1, 1], color='darkorange')
    axes[1, 1].set_title("D: Patient-Wise Adaptivity ($g_v$)", fontweight='bold')
    axes[1, 1].set_xlabel("Average Gate per Patient")

    plt.suptitle("Gated Cross-Attention Audit: Quantifying Non-Saturating Precision Adaptivity", fontsize=16, fontweight='bold', y=0.96)
    plt.savefig("gate_activation_audit_detailed_2x2.png", bbox_inches='tight')
    print("\n💾 Diagnostic Plot saved as 'gate_activation_audit_detailed_2x2.png'")

if __name__ == "__main__":
    audit_mil_pipeline()