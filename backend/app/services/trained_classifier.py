"""Publication-ready ensemble regressor for mutation thermostability prediction.

Predicts DDG (kcal/mol) using an ensemble of 3 models:
  - GradientBoostingRegressor (sklearn)
  - XGBRegressor (xgboost)
  - RandomForestRegressor (sklearn)
Final prediction = average of all 3.

Trained on real experimental data only (no synthetic mutations):
- FireProtDB: ~3,400 curated mutations with DDG values
- ProDDG (S2648): ~2,300 mutations with DDG values
- ThermoMutDB: ~4,000 mutations with DDG values

42 features.
"""

import numpy as np
import pickle
import os
import json

# Path to pre-trained model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "mutation_regressor.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
CONSERVATION_PATH = os.path.join(MODEL_DIR, "conservation_cache.pkl")

_ensemble = None  # dict with 'models' list and 'weights'
_scaler = None
_training_metrics = None
_conservation_cache = None

# ═══════════════════════════════════════════════════════════
# Amino acid property tables
# ═══════════════════════════════════════════════════════════
AA_SET = set("ACDEFGHIKLMNPQRSTVWY")

HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

VOLUME = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6,
}

CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0.5, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}

FLEXIBILITY = {
    'A': 0.36, 'C': 0.35, 'D': 0.51, 'E': 0.50, 'F': 0.31,
    'G': 0.54, 'H': 0.32, 'I': 0.46, 'K': 0.47, 'L': 0.40,
    'M': 0.30, 'N': 0.46, 'P': 0.51, 'Q': 0.49, 'R': 0.53,
    'S': 0.51, 'T': 0.44, 'V': 0.39, 'W': 0.31, 'Y': 0.42,
}

HELIX_PROPENSITY = {
    'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
    'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
    'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
    'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69,
}

SHEET_PROPENSITY = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47,
}

BLOSUM62_DIAG = {
    'A': 4, 'R': 5, 'N': 6, 'D': 6, 'C': 9,
    'Q': 5, 'E': 5, 'G': 6, 'H': 8, 'I': 4,
    'L': 4, 'K': 5, 'M': 5, 'F': 6, 'P': 7,
    'S': 4, 'T': 5, 'W': 11, 'Y': 7, 'V': 4,
}

_BLOSUM62 = {}
_blosum_str = """
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
_lines = [l for l in _blosum_str.strip().split('\n') if l.strip()]
_header = _lines[0].split()
for _line in _lines[1:]:
    _parts = _line.split()
    _aa1 = _parts[0]
    for _j, _aa2 in enumerate(_header):
        _BLOSUM62[(_aa1, _aa2)] = int(_parts[_j + 1])


# ═══════════════════════════════════════════════════════════
# Feature extraction (42 features — matches training script)
# ═══════════════════════════════════════════════════════════

def _estimate_rsa(sequence, position):
    if not sequence or position < 1 or position > len(sequence):
        return 0.5
    idx = position - 1
    aa = sequence[idx]
    h = HYDROPHOBICITY.get(aa, 0)
    base = 0.5 - h * 0.05
    rel_pos = idx / max(len(sequence) - 1, 1)
    if rel_pos < 0.05 or rel_pos > 0.95:
        base += 0.2
    window = sequence[max(0, idx-3):idx+4]
    avg_h = np.mean([HYDROPHOBICITY.get(a, 0) for a in window])
    base -= avg_h * 0.02
    return max(0.0, min(1.0, base))


def _estimate_secondary_structure(sequence, position):
    if not sequence or position < 1 or position > len(sequence):
        return 0.33, 0.33, 0.34
    idx = position - 1
    window = sequence[max(0, idx-4):idx+5]
    h_score = np.mean([HELIX_PROPENSITY.get(a, 1.0) for a in window])
    s_score = np.mean([SHEET_PROPENSITY.get(a, 1.0) for a in window])
    total = h_score + s_score + 1.0
    return h_score / total, s_score / total, 1.0 / total


PSSM_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")

def _get_conservation_features(protein_id, position, wt_aa, mut_aa):
    """Extract 6 PSSM conservation features."""
    if not _conservation_cache or not protein_id:
        return [0.0] * 6
    pssm_data = _conservation_cache.get(protein_id)
    if pssm_data is None:
        return [0.0] * 6
    pssm = pssm_data['pssm']
    info = pssm_data['info_content']
    idx = position - 1
    if idx < 0 or idx >= len(pssm):
        return [0.0] * 6
    aa_to_idx = {aa: i for i, aa in enumerate(PSSM_AA_ORDER)}
    wt_idx = aa_to_idx.get(wt_aa)
    mut_idx = aa_to_idx.get(mut_aa)
    if wt_idx is None or mut_idx is None:
        return [0.0] * 6
    row = pssm[idx]
    pssm_wt = float(row[wt_idx])
    pssm_mut = float(row[mut_idx])
    delta_pssm = pssm_mut - pssm_wt
    info_at_pos = float(info[idx]) if idx < len(info) else 0.0
    rank = float(np.sum(info <= info_at_pos) / max(len(info), 1)) if len(info) > 1 else 0.5
    wt_rank = float(np.sum(row <= pssm_wt) / 20.0)
    return [pssm_wt, pssm_mut, delta_pssm, info_at_pos, rank, wt_rank]


def _extract_features(wt_aa: str, position: int, mut_aa: str,
                      sequence: str = None, protein_id: str = None, **kwargs) -> list[float]:
    """Extract 48 features for a single mutation (42 original + 6 conservation)."""
    if wt_aa not in AA_SET or mut_aa not in AA_SET:
        return [0.0] * 48

    features = []

    # Physicochemical deltas (6)
    dH = HYDROPHOBICITY.get(mut_aa, 0) - HYDROPHOBICITY.get(wt_aa, 0)
    dV = VOLUME.get(mut_aa, 0) - VOLUME.get(wt_aa, 0)
    dC = CHARGE.get(mut_aa, 0) - CHARGE.get(wt_aa, 0)
    dF = FLEXIBILITY.get(mut_aa, 0) - FLEXIBILITY.get(wt_aa, 0)
    dHelix = HELIX_PROPENSITY.get(mut_aa, 1) - HELIX_PROPENSITY.get(wt_aa, 1)
    dSheet = SHEET_PROPENSITY.get(mut_aa, 1) - SHEET_PROPENSITY.get(wt_aa, 1)
    features.extend([dH, dV, dC, dF, dHelix, dSheet])

    # Absolute deltas (6)
    features.extend([abs(dH), abs(dV), abs(dC), abs(dF), abs(dHelix), abs(dSheet)])

    # BLOSUM62 (1)
    features.append(_BLOSUM62.get((wt_aa, mut_aa), 0))

    # Secondary structure at position (3)
    if sequence:
        h, s, c = _estimate_secondary_structure(sequence, position)
    else:
        h, s, c = 0.33, 0.33, 0.34
    features.extend([h, s, c])

    # RSA (1)
    rsa = _estimate_rsa(sequence, position) if sequence else 0.5
    features.append(rsa)

    # Sequence context (4)
    if sequence and 1 <= position <= len(sequence):
        idx = position - 1
        window = sequence[max(0, idx-3):idx+4]
        local_h = np.mean([HYDROPHOBICITY.get(a, 0) for a in window])
        local_c = np.mean([CHARGE.get(a, 0) for a in window])
        gp_count = sum(1 for a in window if a in ('G', 'P'))
        rel_pos = idx / max(len(sequence) - 1, 1)
        features.extend([local_h, local_c, gp_count / len(window), rel_pos])
    else:
        features.extend([0, 0, 0, 0.5])

    # Thermostability features (6)
    to_proline = 1.0 if mut_aa == 'P' and wt_aa != 'P' else 0.0
    from_proline = 1.0 if wt_aa == 'P' and mut_aa != 'P' else 0.0
    to_glycine = 1.0 if mut_aa == 'G' and wt_aa != 'G' else 0.0
    deamid_risk = 0.0
    if wt_aa in ('N', 'Q') and mut_aa not in ('N', 'Q'):
        deamid_risk = -1.0
    elif mut_aa in ('N', 'Q') and wt_aa not in ('N', 'Q'):
        deamid_risk = 1.0
    salt_bridge = 0.0
    if mut_aa in ('D', 'E', 'K', 'R') and wt_aa not in ('D', 'E', 'K', 'R'):
        salt_bridge = 1.0
    elif wt_aa in ('D', 'E', 'K', 'R') and mut_aa not in ('D', 'E', 'K', 'R'):
        salt_bridge = -1.0
    cys_change = 0.0
    if mut_aa == 'C' and wt_aa != 'C':
        cys_change = 1.0
    elif wt_aa == 'C' and mut_aa != 'C':
        cys_change = -1.0
    features.extend([to_proline, from_proline, to_glycine, deamid_risk, salt_bridge, cys_change])

    # Interaction terms (9)
    burial = 1.0 - rsa
    features.extend([
        abs(dH) * burial, abs(dV) * burial, abs(dC) * burial,
        abs(dH) * abs(dV), abs(dC) * abs(dH),
        to_proline * burial, burial * h, burial * s, abs(dH) * h,
    ])

    # Additional (6)
    aromatic_wt = 1.0 if wt_aa in ('F', 'W', 'Y', 'H') else 0.0
    aromatic_mut = 1.0 if mut_aa in ('F', 'W', 'Y', 'H') else 0.0
    small_aa = {'G', 'A', 'S', 'T', 'C'}
    large_aa = {'F', 'W', 'Y', 'R', 'K', 'H'}
    small_to_large = 1.0 if wt_aa in small_aa and mut_aa in large_aa else 0.0
    large_to_small = 1.0 if wt_aa in large_aa and mut_aa in small_aa else 0.0
    cons_wt = BLOSUM62_DIAG.get(wt_aa, 4)
    cons_mut = BLOSUM62_DIAG.get(mut_aa, 4)
    features.extend([
        aromatic_wt - aromatic_mut, small_to_large, large_to_small,
        cons_wt, cons_mut, cons_wt - cons_mut,
    ])

    # PSSM conservation features (6)
    cons_feats = _get_conservation_features(protein_id, position, wt_aa, mut_aa)
    features.extend(cons_feats)

    return features  # 48 features


# ═══════════════════════════════════════════════════════════
# Model loading and prediction
# ═══════════════════════════════════════════════════════════

def train_model(force_retrain: bool = False) -> dict:
    """Load pre-trained ensemble from disk."""
    global _ensemble, _scaler, _training_metrics, _conservation_cache

    if not force_retrain and os.path.exists(REGRESSOR_PATH) and os.path.exists(SCALER_PATH):
        with open(REGRESSOR_PATH, "rb") as f:
            _ensemble = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        # Load conservation cache if available
        if os.path.exists(CONSERVATION_PATH):
            with open(CONSERVATION_PATH, "rb") as f:
                _conservation_cache = pickle.load(f)

        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                _training_metrics = json.load(f)
            _training_metrics["loaded_from_cache"] = True
        else:
            _training_metrics = {
                "model_type": "Ensemble (GradientBoosting + XGBoost + RandomForest)",
                "loaded_from_cache": True,
            }
        return _training_metrics

    raise FileNotFoundError(
        f"Pre-trained model not found at {REGRESSOR_PATH}. "
        "Run train_publication_model.py first."
    )


def _ensemble_predict(features_scaled):
    """Get ensemble prediction (average of all models)."""
    predictions = []
    weights = _ensemble.get('weights', [1.0/3, 1.0/3, 1.0/3])
    for (name, model), w in zip(_ensemble['models'], weights):
        pred = model.predict(features_scaled)
        predictions.append(pred * w)
    return np.sum(predictions, axis=0) / sum(weights)


def predict_mutation(wt_aa: str, position: int, mut_aa: str,
                     sequence: str = None, protein_id: str = None, **kwargs) -> dict:
    """Predict DDG for a mutation using ensemble and derive stability classification."""
    global _ensemble, _scaler

    if _ensemble is None:
        train_model()

    features = np.array([_extract_features(wt_aa, position, mut_aa,
                                           sequence=sequence, protein_id=protein_id)])
    features_scaled = _scaler.transform(features)

    predicted_ddg = float(_ensemble_predict(features_scaled)[0])

    # DDG < 0 means stabilizing (beneficial)
    is_beneficial = predicted_ddg < 0

    # Convert DDG to a probability-like confidence score
    confidence = 1.0 / (1.0 + np.exp(predicted_ddg))  # sigmoid(-ddg)

    return {
        "predicted_beneficial": is_beneficial,
        "predicted_ddg": round(predicted_ddg, 4),
        "confidence": round(float(confidence), 4),
        "probability_beneficial": round(float(confidence), 4),
    }


def predict_candidate_mutations(mutations: list[str], sequence: str = None) -> dict:
    """Predict all mutations in a candidate and return aggregate assessment."""
    if _ensemble is None:
        train_model()

    predictions = []
    total_ddg = 0.0
    for mut_str in mutations:
        wt_aa = mut_str[0]
        mut_aa = mut_str[-1]
        position = int(mut_str[1:-1])

        pred = predict_mutation(wt_aa, position, mut_aa, sequence=sequence)
        pred["mutation"] = mut_str
        predictions.append(pred)
        total_ddg += pred["predicted_ddg"]

    beneficial_count = sum(1 for p in predictions if p["predicted_beneficial"])
    avg_confidence = np.mean([p["confidence"] for p in predictions])

    return {
        "predictions": predictions,
        "all_beneficial": beneficial_count == len(predictions),
        "beneficial_count": beneficial_count,
        "total": len(predictions),
        "total_predicted_ddg": round(total_ddg, 4),
        "average_confidence": round(float(avg_confidence), 4),
    }


def get_training_metrics() -> dict:
    """Return training metrics for display."""
    if _training_metrics is None:
        train_model()
    return _training_metrics
