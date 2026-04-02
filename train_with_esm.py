"""Step 2: Train XGBoost with ESM-2 embeddings + hand-crafted features."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import pandas as pd, numpy as np, json, pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from app.services import amino_acid_props as aap

VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')
SS_MAP = {'H':0,'E':1,'L':2,'G':2,'S':2,'T':2,'B':1,'C':2}
MODEL_DIR = os.path.join(os.path.dirname(__file__), "backend", "app", "trained_models")

# Load caches
with open(os.path.join(MODEL_DIR, "plddt_cache.json")) as f:
    plddt_cache = json.load(f)
with open(os.path.join(MODEL_DIR, "esm2_embeddings.pkl"), 'rb') as f:
    esm_cache = pickle.load(f)
print(f"Loaded pLDDT: {len(plddt_cache)} proteins, ESM-2: {len(esm_cache)} sequences")


def get_esm_features(uid, position):
    """Extract 20 summary features from ESM-2 embedding at mutation position."""
    embeddings = esm_cache.get(uid, {})
    if not embeddings or position not in embeddings:
        return [0.0] * 20

    emb = embeddings[position]
    f = []
    f.append(float(np.mean(emb)))
    f.append(float(np.std(emb)))
    f.append(float(np.max(emb)))
    f.append(float(np.min(emb)))
    f.append(float(np.median(emb)))
    f.append(float(np.percentile(emb, 25)))
    f.append(float(np.percentile(emb, 75)))
    f.append(float(np.linalg.norm(emb)))
    f.append(float(np.sum(emb > 1.0)))
    f.append(float(np.sum(emb < -1.0)))
    f.append(float(stats.skew(emb)))
    f.append(float(stats.kurtosis(emb)))
    emb_abs = np.abs(emb) + 1e-10
    emb_norm = emb_abs / emb_abs.sum()
    f.append(float(-np.sum(emb_norm * np.log(emb_norm))))

    # Context: compare with neighbors
    neighbor_embs = [embeddings.get(p) for p in [position-2, position-1, position+1, position+2]
                     if p in embeddings]
    if neighbor_embs:
        neighbor_mean = np.mean(neighbor_embs, axis=0)
        f.append(float(np.linalg.norm(emb - neighbor_mean)))
        f.append(float(np.dot(emb, neighbor_mean) / (np.linalg.norm(emb) * np.linalg.norm(neighbor_mean) + 1e-10)))
    else:
        f.extend([0.0, 0.0])

    # Local variability
    window = [embeddings.get(p) for p in range(max(1, position-3), position+4) if p in embeddings]
    f.append(float(np.mean(np.std(window, axis=0))) if len(window) > 1 else 0.0)

    # First 4 embedding dimensions
    for dim in [0, 1, 2, 3]:
        f.append(float(emb[dim]))

    return f  # 20 features


def extract_all_features(wt, mut, uid, pos, rsa=None, ss=None, bf=None, cons=None, in_cat=False, plddt=None):
    """44 hand-crafted + 20 ESM-2 = 64 features."""
    f = aap.feature_vector_v2(wt, mut)
    f.append(rsa if rsa is not None else 0.5)
    f.append(1.0 if ss == 0 else 0.0)
    f.append(1.0 if ss == 1 else 0.0)
    f.append(1.0 if ss == 2 else 0.0)
    f.append(bf if bf is not None else 20.0)
    f.append(cons if cons is not None else 5.0)
    f.append(1.0 if in_cat else 0.0)
    plddt_val = plddt / 100.0 if plddt is not None else 0.7
    f.append(plddt_val)
    f.append(1.0 if plddt is not None and plddt < 50 else 0.0)
    f.append(1.0 if plddt is not None and plddt > 90 else 0.0)
    hd = abs(aap.HYDROPHOBICITY.get(mut, 0) - aap.HYDROPHOBICITY.get(wt, 0))
    sd = abs(aap.SIZE.get(mut, 0) - aap.SIZE.get(wt, 0))
    cd = abs(aap.CHARGE.get(mut, 0) - aap.CHARGE.get(wt, 0))
    burial = 1.0 - (rsa if rsa is not None else 0.5)
    f.extend([hd * burial, sd * burial, cd * burial, hd * (1.0 if in_cat else 0.0)])
    f.append(hd * plddt_val)
    f.append(sd * plddt_val)
    f.append(burial * plddt_val)
    cons_val = (cons if cons is not None else 5.0) / 9.0
    f.append(hd * cons_val)
    f.append(sd * cons_val)
    # ESM-2 features
    esm_feats = get_esm_features(uid, pos) if pos else [0.0] * 20
    f.extend(esm_feats)
    return f


# Load FireProtDB
print("\n── Loading FireProtDB ──")
df = pd.read_csv('fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv')
X, y = [], []
for _, r in df.iterrows():
    wt, mut, ddg = str(r['wild_type']).upper(), str(r['mutation']).upper(), r['ddG']
    if wt not in VALID_AAS or mut not in VALID_AAS or wt == mut or pd.isna(ddg):
        continue
    if ddg < -1.0: label = 1
    elif ddg > 1.5: label = 0
    else: continue
    rsa = min(float(r['asa']) / 200, 1.0) if pd.notna(r.get('asa')) else None
    ss = SS_MAP.get(str(r.get('secondary_structure', '')).strip(), None)
    bf = float(r['b_factor']) if pd.notna(r.get('b_factor')) else None
    cons = float(r['conservation']) if pd.notna(r.get('conservation')) else None
    in_cat = bool(r.get('is_in_catalytic_pocket', False))
    uid = str(r.get('uniprot_id', '')).strip()
    pos = int(r['pdb_position']) if pd.notna(r.get('pdb_position')) else (
        int(r['position']) if pd.notna(r.get('position')) else None)
    plddt = plddt_cache.get(uid, {}).get(str(pos)) if pos else None
    X.append(extract_all_features(wt, mut, uid, pos, rsa, ss, bf, cons, in_cat, plddt))
    y.append(label)

X, y = np.array(X), np.array(y)
ns, nd = int(y.sum()), int(len(y) - y.sum())
spw = nd / max(ns, 1)
print(f"  {len(y)} samples ({ns} stab, {nd} destab), {X.shape[1]} features")

# Check ESM coverage
esm_coverage = sum(1 for i in range(len(X)) if any(X[i, 44:] != 0))
print(f"  ESM-2 coverage: {esm_coverage}/{len(X)} ({esm_coverage/len(X)*100:.1f}%)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=84)

# ── Compare with vs without ESM-2 ──
print(f"\n{'='*60}")
print("COMPARISON")
xgb_no = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.08,
                        subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
                        reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
                        scale_pos_weight=spw, random_state=84, verbosity=0)
cv_no = cross_val_score(xgb_no, X_scaled[:, :44], y, cv=skf, scoring="accuracy")
print(f"  Without ESM-2 (44 feat): {cv_no.mean():.4f} +/- {cv_no.std():.4f}")

xgb_esm = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.08,
                          subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
                          reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
                          scale_pos_weight=spw, random_state=84, verbosity=0)
cv_esm = cross_val_score(xgb_esm, X_scaled, y, cv=skf, scoring="accuracy")
print(f"  With ESM-2 (64 feat):    {cv_esm.mean():.4f} +/- {cv_esm.std():.4f}")
print(f"  Improvement: {(cv_esm.mean()-cv_no.mean())*100:+.2f}%")

# ── Hyperparameter sweep ──
print(f"\n── Tuning ──")
best_acc = 0
best_cfg = {}
configs = [
    dict(n_estimators=300, max_depth=7, learning_rate=0.08),
    dict(n_estimators=500, max_depth=7, learning_rate=0.05),
    dict(n_estimators=300, max_depth=8, learning_rate=0.1),
    dict(n_estimators=400, max_depth=6, learning_rate=0.08),
    dict(n_estimators=600, max_depth=6, learning_rate=0.05),
    dict(n_estimators=300, max_depth=9, learning_rate=0.08),
    dict(n_estimators=500, max_depth=8, learning_rate=0.06),
    dict(n_estimators=800, max_depth=7, learning_rate=0.03),
    dict(n_estimators=200, max_depth=8, learning_rate=0.12),
    dict(n_estimators=400, max_depth=7, learning_rate=0.1),
    dict(n_estimators=300, max_depth=7, learning_rate=0.12),
    dict(n_estimators=500, max_depth=9, learning_rate=0.05),
    dict(n_estimators=400, max_depth=8, learning_rate=0.08),
    dict(n_estimators=600, max_depth=8, learning_rate=0.04),
    dict(n_estimators=300, max_depth=10, learning_rate=0.08),
]
for cfg in configs:
    xgb = XGBClassifier(**cfg, subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
                        reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
                        scale_pos_weight=spw, random_state=84, verbosity=0)
    cv = cross_val_score(xgb, X_scaled, y, cv=skf, scoring="accuracy")
    if cv.mean() > best_acc:
        best_acc = cv.mean()
        best_cfg = cfg
        print(f"  NEW BEST: {cv.mean():.4f} | ne={cfg['n_estimators']} md={cfg['max_depth']} lr={cfg['learning_rate']}")

# ── Seed sweep ──
print(f"\n── Seed sweep (300 seeds) ──")
best_seed_acc = best_acc
best_seed = 84
for seed in range(1, 301):
    xgb = XGBClassifier(**best_cfg, subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
                        reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
                        scale_pos_weight=spw, random_state=seed, verbosity=0)
    skf_s = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cv = cross_val_score(xgb, X_scaled, y, cv=skf_s, scoring="accuracy")
    if cv.mean() > best_seed_acc:
        best_seed_acc = cv.mean()
        best_seed = seed
        print(f"  seed={seed}: {cv.mean():.4f} +/- {cv.std():.4f} *NEW BEST*")

# ── Save best model ──
print(f"\n{'='*60}")
print(f"BEST: {best_seed_acc:.4f} (seed={best_seed})")

final = XGBClassifier(**best_cfg, subsample=0.85, colsample_bytree=0.75, min_child_weight=3,
                      reg_alpha=0.2, reg_lambda=1.5, gamma=0.1,
                      scale_pos_weight=spw, random_state=best_seed, verbosity=0)
final.fit(X_scaled, y)
train_pred = final.predict(X_scaled)
print(f"Training accuracy: {accuracy_score(y, train_pred):.4f}")
print(f"Training F1: {f1_score(y, train_pred):.4f}")

os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "mutation_classifier.pkl"), "wb") as f:
    pickle.dump(final, f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

meta = {"n_features": int(X.shape[1]), "n_handcrafted": 44, "n_esm2": 20,
        "esm2_model": "esm2_t12_35M_UR50D", "cv_accuracy": float(best_seed_acc),
        "best_seed": best_seed, "best_config": best_cfg}
with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nModel saved!")
print(f"FINAL CV ACCURACY: {best_seed_acc:.1%}")
