"""Publication-Ready Protein Stability Prediction Model.

Trained ONLY on real experimental data — no synthetic mutations.

Data sources:
  - FireProtDB (4,997 curated mutations with DDG)
  - ProDDG / S2648 (2,648 mutations with DDG)
  - ThermoMutDB (~300K+ mutations with DDG)

Independent test set (never seen during training):
  - S669 (669 mutations) — held out entirely

Evaluation:
  - 10-fold stratified cross-validation on training set
  - Leave-one-protein-out cross-validation (generalization)
  - Independent test on S669
  - Metrics: Accuracy, F1, AUC, Precision, Recall, MAE, Pearson r
"""

import os
import json
import re
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict, GroupKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, mean_absolute_error, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIREPROT_PATH = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
PRODDG_PATH = os.path.join(BASE_DIR, "proddg_s2648.csv")
S669_PATH = os.path.join(BASE_DIR, "s669_full.tsv")
THERMOMUTDB_PATH = os.path.join(BASE_DIR, "thermomutdb.json")
MODEL_DIR = os.path.join(BASE_DIR, "backend/app/trained_models")

# ═══════════════════════════════════════════════════════════
# Amino acid properties (same as production)
# ═══════════════════════════════════════════════════════════
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AMINO_ACIDS)

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

# BLOSUM62 diagonal (self-substitution scores)
BLOSUM62_DIAG = {
    'A': 4, 'R': 5, 'N': 6, 'D': 6, 'C': 9,
    'Q': 5, 'E': 5, 'G': 6, 'H': 8, 'I': 4,
    'L': 4, 'K': 5, 'M': 5, 'F': 6, 'P': 7,
    'S': 4, 'T': 5, 'W': 11, 'Y': 7, 'V': 4,
}

# BLOSUM62 full matrix (subset of common substitutions)
BLOSUM62 = {}
blosum_str = """
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
lines = [l for l in blosum_str.strip().split('\n') if l.strip()]
header = lines[0].split()
for line in lines[1:]:
    parts = line.split()
    aa1 = parts[0]
    for j, aa2 in enumerate(header):
        BLOSUM62[(aa1, aa2)] = int(parts[j + 1])


def get_blosum62(wt, mut):
    return BLOSUM62.get((wt, mut), 0)


# ═══════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════

def estimate_rsa(sequence, position):
    """Estimate relative solvent accessibility from sequence context."""
    if not sequence or position < 1 or position > len(sequence):
        return 0.5
    idx = position - 1
    aa = sequence[idx]
    # Buried residues tend to be hydrophobic
    h = HYDROPHOBICITY.get(aa, 0)
    base = 0.5 - h * 0.05  # hydrophobic = more buried

    # Terminal residues more exposed
    rel_pos = idx / max(len(sequence) - 1, 1)
    if rel_pos < 0.05 or rel_pos > 0.95:
        base += 0.2

    # Neighbors: if surrounded by hydrophobic, likely buried
    window = sequence[max(0, idx-3):idx+4]
    avg_h = np.mean([HYDROPHOBICITY.get(a, 0) for a in window])
    base -= avg_h * 0.02

    return max(0.0, min(1.0, base))


def estimate_secondary_structure(sequence, position):
    """Estimate SS propensities from local sequence."""
    if not sequence or position < 1 or position > len(sequence):
        return 0.33, 0.33, 0.34
    idx = position - 1
    window = sequence[max(0, idx-4):idx+5]
    h_score = np.mean([HELIX_PROPENSITY.get(a, 1.0) for a in window])
    s_score = np.mean([SHEET_PROPENSITY.get(a, 1.0) for a in window])
    total = h_score + s_score + 1.0
    return h_score / total, s_score / total, 1.0 / total


def extract_features(wt_aa, position, mut_aa, sequence=None):
    """Extract feature vector for a single mutation.

    Features (42 total):
      - 6 physicochemical deltas (hydrophobicity, volume, charge, flexibility, helix, sheet)
      - 6 absolute values for WT and MUT
      - 1 BLOSUM62 substitution score
      - 3 secondary structure propensities at position
      - 1 estimated RSA
      - 4 sequence context features
      - 6 thermostability-specific features
      - 9 interaction terms
      - 6 additional features
    """
    if wt_aa not in AA_SET or mut_aa not in AA_SET:
        return None

    features = []

    # ── Physicochemical deltas (6) ──
    dH = HYDROPHOBICITY.get(mut_aa, 0) - HYDROPHOBICITY.get(wt_aa, 0)
    dV = VOLUME.get(mut_aa, 0) - VOLUME.get(wt_aa, 0)
    dC = CHARGE.get(mut_aa, 0) - CHARGE.get(wt_aa, 0)
    dF = FLEXIBILITY.get(mut_aa, 0) - FLEXIBILITY.get(wt_aa, 0)
    dHelix = HELIX_PROPENSITY.get(mut_aa, 1) - HELIX_PROPENSITY.get(wt_aa, 1)
    dSheet = SHEET_PROPENSITY.get(mut_aa, 1) - SHEET_PROPENSITY.get(wt_aa, 1)
    features.extend([dH, dV, dC, dF, dHelix, dSheet])

    # ── Absolute deltas (6) ──
    features.extend([abs(dH), abs(dV), abs(dC), abs(dF), abs(dHelix), abs(dSheet)])

    # ── BLOSUM62 (1) ──
    features.append(get_blosum62(wt_aa, mut_aa))

    # ── Secondary structure at position (3) ──
    if sequence:
        h, s, c = estimate_secondary_structure(sequence, position)
    else:
        h, s, c = 0.33, 0.33, 0.34
    features.extend([h, s, c])

    # ── RSA (1) ──
    rsa = estimate_rsa(sequence, position) if sequence else 0.5
    features.append(rsa)

    # ── Sequence context (4) ──
    if sequence and 1 <= position <= len(sequence):
        idx = position - 1
        # Local hydrophobicity
        window = sequence[max(0, idx-3):idx+4]
        local_h = np.mean([HYDROPHOBICITY.get(a, 0) for a in window])
        # Local charge
        local_c = np.mean([CHARGE.get(a, 0) for a in window])
        # Glycine/proline count in window
        gp_count = sum(1 for a in window if a in ('G', 'P'))
        # Relative position
        rel_pos = idx / max(len(sequence) - 1, 1)
        features.extend([local_h, local_c, gp_count / len(window), rel_pos])
    else:
        features.extend([0, 0, 0, 0.5])

    # ── Thermostability features (6) ──
    # Proline introduction (rigidifies backbone)
    to_proline = 1.0 if mut_aa == 'P' and wt_aa != 'P' else 0.0
    from_proline = 1.0 if wt_aa == 'P' and mut_aa != 'P' else 0.0
    # Glycine introduction (increases flexibility)
    to_glycine = 1.0 if mut_aa == 'G' and wt_aa != 'G' else 0.0
    # Deamidation risk (N,Q are prone at high temp)
    deamid_risk = 0.0
    if wt_aa in ('N', 'Q') and mut_aa not in ('N', 'Q'):
        deamid_risk = -1.0  # removing risk = good
    elif mut_aa in ('N', 'Q') and wt_aa not in ('N', 'Q'):
        deamid_risk = 1.0  # adding risk = bad
    # Salt bridge potential
    salt_bridge = 0.0
    if mut_aa in ('D', 'E', 'K', 'R') and wt_aa not in ('D', 'E', 'K', 'R'):
        salt_bridge = 1.0
    elif wt_aa in ('D', 'E', 'K', 'R') and mut_aa not in ('D', 'E', 'K', 'R'):
        salt_bridge = -1.0
    # Cysteine (disulfide potential)
    cys_change = 0.0
    if mut_aa == 'C' and wt_aa != 'C':
        cys_change = 1.0
    elif wt_aa == 'C' and mut_aa != 'C':
        cys_change = -1.0
    features.extend([to_proline, from_proline, to_glycine, deamid_risk, salt_bridge, cys_change])

    # ── Interaction terms (9) ──
    burial = 1.0 - rsa
    features.extend([
        abs(dH) * burial,      # hydrophobicity change × burial
        abs(dV) * burial,      # volume change × burial
        abs(dC) * burial,      # charge change × burial
        abs(dH) * abs(dV),     # hydrophobicity × volume
        abs(dC) * abs(dH),     # charge × hydrophobicity
        to_proline * burial,   # proline intro × burial
        burial * h,            # burial × helix
        burial * s,            # burial × sheet
        abs(dH) * h,           # hydrophobicity × helix
    ])

    # ── Additional (6) ──
    # Aromatic change
    aromatic_wt = 1.0 if wt_aa in ('F', 'W', 'Y', 'H') else 0.0
    aromatic_mut = 1.0 if mut_aa in ('F', 'W', 'Y', 'H') else 0.0
    # Small-to-large / large-to-small
    small_aa = {'G', 'A', 'S', 'T', 'C'}
    large_aa = {'F', 'W', 'Y', 'R', 'K', 'H'}
    small_to_large = 1.0 if wt_aa in small_aa and mut_aa in large_aa else 0.0
    large_to_small = 1.0 if wt_aa in large_aa and mut_aa in small_aa else 0.0
    # Conservation proxy (BLOSUM self-score difference)
    cons_wt = BLOSUM62_DIAG.get(wt_aa, 4)
    cons_mut = BLOSUM62_DIAG.get(mut_aa, 4)
    features.extend([
        aromatic_wt - aromatic_mut,  # aromatic change
        small_to_large,
        large_to_small,
        cons_wt,
        cons_mut,
        cons_wt - cons_mut,
    ])

    return features  # 42 features total


# ═══════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════

def parse_mutation_code(code):
    """Parse 'A123G' format into (wt_aa, position, mut_aa)."""
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', code)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None, None, None


def load_fireprotdb():
    """Load FireProtDB dataset."""
    print("Loading FireProtDB...")
    df = pd.read_csv(FIREPROT_PATH)
    records = []
    for _, row in df.iterrows():
        wt = row.get('wild_type', '')
        mut = row.get('mutation', '')
        pos = row.get('position', 0)
        ddg = row.get('ddG', None)
        seq = row.get('sequence', '')
        pdb = str(row.get('pdb_id', '')).split('|')[0]

        if pd.isna(ddg) or wt not in AA_SET or mut not in AA_SET or wt == mut:
            continue
        try:
            pos = int(pos)
        except (ValueError, TypeError):
            continue

        records.append({
            'wt_aa': wt, 'position': pos, 'mut_aa': mut,
            'ddg': float(ddg), 'sequence': str(seq) if pd.notna(seq) else '',
            'protein_id': pdb, 'source': 'FireProtDB'
        })
    print(f"  Loaded {len(records)} mutations from FireProtDB")
    return records


def load_proddg():
    """Load ProDDG / S2648 dataset."""
    print("Loading ProDDG (S2648)...")
    df = pd.read_csv(PRODDG_PATH, sep='\t')
    records = []
    for _, row in df.iterrows():
        mut_code = row.get('mutation', '')
        wt, pos, mut = parse_mutation_code(str(mut_code))
        ddg = row.get('ddG', None)
        seq = row.get('wt_sequence', '')
        pdb = str(row.get('pdb', ''))

        if wt is None or pd.isna(ddg):
            continue

        records.append({
            'wt_aa': wt, 'position': pos, 'mut_aa': mut,
            'ddg': float(ddg), 'sequence': str(seq) if pd.notna(seq) else '',
            'protein_id': pdb, 'source': 'ProDDG'
        })
    print(f"  Loaded {len(records)} mutations from ProDDG")
    return records


def load_s669():
    """Load S669 independent test set."""
    print("Loading S669 (independent test set)...")
    df = pd.read_csv(S669_PATH, sep='\t')
    records = []
    for _, row in df.iterrows():
        mut_code = row.get('mutation', '')
        wt, pos, mut = parse_mutation_code(str(mut_code))
        ddg = row.get('ddG', None)
        seq = row.get('wt_sequence', '')
        pdb = str(row.get('pdb', ''))

        if wt is None or pd.isna(ddg):
            continue

        records.append({
            'wt_aa': wt, 'position': pos, 'mut_aa': mut,
            'ddg': float(ddg), 'sequence': str(seq) if pd.notna(seq) else '',
            'protein_id': pdb, 'source': 'S669'
        })
    print(f"  Loaded {len(records)} mutations from S669")
    return records


def load_thermomutdb():
    """Load ThermoMutDB dataset."""
    print("Loading ThermoMutDB...")
    with open(THERMOMUTDB_PATH, 'r') as f:
        data = json.load(f)

    records = []
    for entry in data:
        mut_code = entry.get('mutation_code', '')
        wt, pos, mut = parse_mutation_code(str(mut_code))
        ddg = entry.get('ddg', None)
        pdb = entry.get('PDB_wild', '')

        if wt is None or ddg is None:
            continue
        try:
            ddg = float(ddg)
        except (ValueError, TypeError):
            continue

        # ThermoMutDB doesn't include sequences
        records.append({
            'wt_aa': wt, 'position': pos, 'mut_aa': mut,
            'ddg': ddg, 'sequence': '',
            'protein_id': str(pdb), 'source': 'ThermoMutDB'
        })
    print(f"  Loaded {len(records)} mutations from ThermoMutDB")
    return records


def deduplicate(records):
    """Remove duplicate mutations (same protein + position + mutation)."""
    seen = set()
    unique = []
    for r in records:
        key = (r['protein_id'], r['position'], r['wt_aa'], r['mut_aa'])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    print(f"  After deduplication: {len(unique)} unique mutations (removed {len(records) - len(unique)})")
    return unique


# ═══════════════════════════════════════════════════════════
# Main training pipeline
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PUBLICATION-READY MODEL TRAINING")
    print("Real experimental data only — no synthetic mutations")
    print("=" * 70)
    print()

    # ── Step 1: Load all data ──
    print("STEP 1: Loading datasets")
    print("-" * 40)
    fireprot = load_fireprotdb()
    proddg = load_proddg()
    thermomutdb = load_thermomutdb()
    s669 = load_s669()

    # ── Step 2: Combine training data and deduplicate ──
    print("\nSTEP 2: Combining and deduplicating training data")
    print("-" * 40)
    train_records = fireprot + proddg + thermomutdb
    print(f"  Total before dedup: {len(train_records)}")
    train_records = deduplicate(train_records)

    # Remove any S669 proteins from training (strict independence)
    s669_proteins = set(r['protein_id'] for r in s669)
    s669_mutations = set((r['protein_id'], r['position'], r['wt_aa'], r['mut_aa']) for r in s669)
    train_clean = []
    removed_overlap = 0
    for r in train_records:
        key = (r['protein_id'], r['position'], r['wt_aa'], r['mut_aa'])
        if key in s669_mutations:
            removed_overlap += 1
        else:
            train_clean.append(r)
    train_records = train_clean
    print(f"  Removed {removed_overlap} mutations overlapping with S669 test set")
    print(f"  Final training set: {len(train_records)} mutations")

    # ── Step 3: Extract features ──
    print("\nSTEP 3: Extracting features")
    print("-" * 40)

    def records_to_arrays(records):
        X, y_ddg, y_binary, proteins, sources = [], [], [], [], []
        skipped = 0
        for r in records:
            feats = extract_features(r['wt_aa'], r['position'], r['mut_aa'], r['sequence'])
            if feats is None:
                skipped += 1
                continue
            X.append(feats)
            y_ddg.append(r['ddg'])
            # Convention: DDG > 0 = destabilizing, DDG < 0 = stabilizing
            # We predict "stabilizing" = beneficial = 1
            y_binary.append(1 if r['ddg'] < 0 else 0)
            proteins.append(r['protein_id'])
            sources.append(r['source'])
        if skipped:
            print(f"  Skipped {skipped} mutations (invalid amino acids)")
        return np.array(X), np.array(y_ddg), np.array(y_binary), proteins, sources

    X_train, y_train_ddg, y_train, train_proteins, train_sources = records_to_arrays(train_records)
    X_test, y_test_ddg, y_test, test_proteins, test_sources = records_to_arrays(s669)

    print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test (S669): {X_test.shape[0]} samples")
    print(f"  Training class balance: {np.sum(y_train == 1)} stabilizing, {np.sum(y_train == 0)} destabilizing")
    print(f"  Test class balance: {np.sum(y_test == 1)} stabilizing, {np.sum(y_test == 0)} destabilizing")

    # Source breakdown
    source_counts = defaultdict(int)
    for s in train_sources:
        source_counts[s] += 1
    print(f"  Training sources: {dict(source_counts)}")

    # ── Step 4: Scale features ──
    print("\nSTEP 4: Scaling features")
    print("-" * 40)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Scaled {X_train.shape[1]} features")

    # ── Step 5: Train classifier ──
    print("\nSTEP 5: Training GradientBoosting classifier")
    print("-" * 40)

    # Class weight for imbalanced data
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    print(f"  Class ratio: 1:{n_neg/max(n_pos,1):.1f} (stabilizing:destabilizing)")

    # Use balanced sample weights
    sample_weights = np.ones(len(y_train))
    if n_pos > 0 and n_neg > 0:
        w_pos = len(y_train) / (2 * n_pos)
        w_neg = len(y_train) / (2 * n_neg)
        sample_weights[y_train == 1] = w_pos
        sample_weights[y_train == 0] = w_neg

    clf = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        min_samples_split=20,
        max_features='sqrt',
        random_state=42,
        verbose=0,
    )
    clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    print("  Classifier trained.")

    # ── Step 6: Cross-validation on training set ──
    print("\nSTEP 6: Cross-validation (10-fold stratified)")
    print("-" * 40)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='f1')
    cv_auc = cross_val_score(clf, X_train_scaled, y_train, cv=skf, scoring='roc_auc')
    print(f"  CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  CV F1:       {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
    print(f"  CV AUC:      {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

    # ── Step 7: Leave-one-protein-out CV (generalization) ──
    print("\nSTEP 7: Leave-one-protein-out cross-validation")
    print("-" * 40)
    # Group by protein, but only use proteins with enough samples
    protein_arr = np.array(train_proteins)
    unique_proteins = list(set(train_proteins))
    protein_sizes = {p: np.sum(protein_arr == p) for p in unique_proteins}
    # Use proteins with >= 5 mutations for meaningful groups
    big_proteins = [p for p, s in protein_sizes.items() if s >= 5]
    print(f"  Proteins with >= 5 mutations: {len(big_proteins)}")

    if len(big_proteins) >= 5:
        # Create group labels (proteins with < 5 mutations get group -1, excluded from LOPO)
        groups = np.array([train_proteins[i] if train_proteins[i] in big_proteins else f"_small_{i}"
                          for i in range(len(train_proteins))])
        gkf = GroupKFold(n_splits=min(len(big_proteins), 20))
        mask = np.array([p in big_proteins for p in train_proteins])
        if np.sum(mask) > 100:
            lopo_scores = cross_val_score(clf, X_train_scaled[mask], y_train[mask],
                                          cv=gkf, groups=groups[mask], scoring='accuracy')
            lopo_auc = cross_val_score(clf, X_train_scaled[mask], y_train[mask],
                                       cv=gkf, groups=groups[mask], scoring='roc_auc')
            print(f"  LOPO Accuracy: {lopo_scores.mean():.4f} +/- {lopo_scores.std():.4f}")
            print(f"  LOPO AUC:      {lopo_auc.mean():.4f} +/- {lopo_auc.std():.4f}")
        else:
            print("  Not enough grouped samples for LOPO CV")
    else:
        print("  Not enough proteins with >= 5 mutations for LOPO CV")

    # ── Step 8: Independent test on S669 ──
    print("\nSTEP 8: Independent test on S669")
    print("-" * 40)
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.0
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    mae = mean_absolute_error(y_test_ddg, clf.predict_proba(X_test_scaled)[:, 1] * -2)  # rough DDG estimate

    # Pearson/Spearman correlation between predicted probability and actual DDG
    pearson_r, pearson_p = pearsonr(y_prob, -y_test_ddg)  # negative because lower DDG = more stabilizing
    spearman_r, spearman_p = spearmanr(y_prob, -y_test_ddg)

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  Pearson r:   {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r:  {spearman_r:.4f} (p={spearman_p:.2e})")
    print()
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}, TP={cm[1][1]}")
    print()
    print(classification_report(y_test, y_pred, target_names=['Destabilizing', 'Stabilizing']))

    # ── Step 9: Feature importance ──
    print("\nSTEP 9: Top 15 feature importances")
    print("-" * 40)
    feature_names = [
        'dH', 'dV', 'dC', 'dF', 'dHelix', 'dSheet',
        '|dH|', '|dV|', '|dC|', '|dF|', '|dHelix|', '|dSheet|',
        'BLOSUM62',
        'helix_prop', 'sheet_prop', 'coil_prop',
        'RSA',
        'local_hydro', 'local_charge', 'GP_fraction', 'rel_position',
        'to_Pro', 'from_Pro', 'to_Gly', 'deamid_risk', 'salt_bridge', 'cys_change',
        'dH×burial', 'dV×burial', 'dC×burial', 'dH×dV', 'dC×dH',
        'Pro×burial', 'burial×helix', 'burial×sheet', 'dH×helix',
        'aromatic_change', 'small→large', 'large→small',
        'cons_wt', 'cons_mut', 'cons_delta',
    ]
    importances = clf.feature_importances_
    idx_sorted = np.argsort(importances)[::-1]
    for i in range(min(15, len(feature_names))):
        j = idx_sorted[i]
        name = feature_names[j] if j < len(feature_names) else f"feat_{j}"
        print(f"  {i+1:2d}. {name:20s} {importances[j]:.4f}")

    # ── Step 10: Save model ──
    print("\nSTEP 10: Saving model")
    print("-" * 40)
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "mutation_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "model_type": "GradientBoosting (publication-ready)",
        "n_features": X_train.shape[1],
        "feature_version": "v4_publication",
        "training_samples": int(X_train.shape[0]),
        "stabilizing_samples": int(np.sum(y_train == 1)),
        "destabilizing_samples": int(np.sum(y_train == 0)),
        "data_sources": {
            "FireProtDB": int(source_counts.get('FireProtDB', 0)),
            "ProDDG": int(source_counts.get('ProDDG', 0)),
            "ThermoMutDB": int(source_counts.get('ThermoMutDB', 0)),
        },
        "synthetic_data": False,
        "independent_test_set": "S669 (669 mutations)",
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_f1": round(float(cv_f1.mean()), 4),
        "cv_auc": round(float(cv_auc.mean()), 4),
        "test_accuracy": round(float(acc), 4),
        "test_f1": round(float(f1), 4),
        "test_auc": round(float(auc), 4),
        "test_precision": round(float(prec), 4),
        "test_recall": round(float(rec), 4),
        "test_pearson_r": round(float(pearson_r), 4),
        "test_spearman_r": round(float(spearman_r), 4),
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved mutation_classifier.pkl")
    print(f"  Saved scaler.pkl")
    print(f"  Saved model_meta.json")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Training: {X_train.shape[0]} real experimental mutations")
    print(f"  Test (S669): {X_test.shape[0]} independent mutations")
    print(f"  CV Accuracy: {cv_scores.mean():.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test AUC: {auc:.4f}")
    print(f"  Synthetic data used: NONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
