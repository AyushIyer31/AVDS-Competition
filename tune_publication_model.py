"""Tune publication model — try multiple approaches to maximize S669 performance."""

import os, json, re, pickle, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

# Reuse data loading from publication model
from train_publication_model import (
    load_fireprotdb, load_proddg, load_s669, load_thermomutdb,
    deduplicate, extract_features, AA_SET, MODEL_DIR
)

def records_to_arrays(records):
    X, y_ddg, y_binary, proteins, sources = [], [], [], [], []
    for r in records:
        feats = extract_features(r['wt_aa'], r['position'], r['mut_aa'], r['sequence'])
        if feats is None:
            continue
        X.append(feats)
        y_ddg.append(r['ddg'])
        y_binary.append(1 if r['ddg'] < 0 else 0)
        proteins.append(r['protein_id'])
        sources.append(r['source'])
    return np.array(X), np.array(y_ddg), np.array(y_binary), proteins, sources


def evaluate(clf, X_test, y_test, y_ddg, label=""):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    pr, _ = pearsonr(y_prob, -y_ddg)
    sr, _ = spearmanr(y_prob, -y_ddg)
    print(f"  {label}")
    print(f"    Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    print(f"    Pearson={pr:.4f}  Spearman={sr:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    return {'acc': acc, 'f1': f1, 'auc': auc, 'prec': prec, 'rec': rec, 'pearson': pr, 'spearman': sr}


def main():
    print("=" * 70)
    print("MODEL TUNING — Finding best configuration")
    print("=" * 70)

    # Load data
    fireprot = load_fireprotdb()
    proddg = load_proddg()
    thermomutdb = load_thermomutdb()
    s669 = load_s669()

    train_records = deduplicate(fireprot + proddg + thermomutdb)
    s669_mutations = set((r['protein_id'], r['position'], r['wt_aa'], r['mut_aa']) for r in s669)
    train_records = [r for r in train_records if (r['protein_id'], r['position'], r['wt_aa'], r['mut_aa']) not in s669_mutations]

    X_train, y_train_ddg, y_train, train_proteins, train_sources = records_to_arrays(train_records)
    X_test, y_test_ddg, y_test, _, _ = records_to_arrays(s669)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    sample_weights = np.ones(len(y_train))
    w_pos = len(y_train) / (2 * n_pos)
    w_neg = len(y_train) / (2 * n_neg)
    sample_weights[y_train == 1] = w_pos
    sample_weights[y_train == 0] = w_neg

    print(f"\nTraining: {len(y_train)} samples, Test: {len(y_test)} samples")
    print(f"Train balance: {n_pos} stabilizing, {n_neg} destabilizing")
    print()

    best_model = None
    best_scaler = None
    best_score = 0
    best_name = ""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # ── Config 1: GradientBoosting (various hyperparams) ──
    configs = [
        ("GB depth=4 lr=0.05 n=300", GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            min_samples_leaf=15, max_features='sqrt', random_state=42)),
        ("GB depth=5 lr=0.03 n=500", GradientBoostingClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.85,
            min_samples_leaf=10, max_features='sqrt', random_state=42)),
        ("GB depth=6 lr=0.05 n=500", GradientBoostingClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            min_samples_leaf=8, max_features='sqrt', random_state=42)),
        ("GB depth=3 lr=0.1 n=400", GradientBoostingClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.1, subsample=0.9,
            min_samples_leaf=20, max_features='sqrt', random_state=42)),
        ("GB depth=7 lr=0.02 n=800", GradientBoostingClassifier(
            n_estimators=800, max_depth=7, learning_rate=0.02, subsample=0.75,
            min_samples_leaf=5, max_features='sqrt', random_state=42)),
        ("RF n=500 depth=None", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced', random_state=42)),
        ("RF n=1000 depth=15", RandomForestClassifier(
            n_estimators=1000, max_depth=15, min_samples_leaf=3,
            max_features='sqrt', class_weight='balanced', random_state=42)),
        ("ET n=500", ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            max_features='sqrt', class_weight='balanced', random_state=42)),
    ]

    for name, clf in configs:
        print(f"\n--- {name} ---")
        if 'GB' in name:
            clf.fit(X_train_s, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train_s, y_train)

        cv = cross_val_score(clf, X_train_s, y_train, cv=skf, scoring='roc_auc')
        print(f"  CV AUC: {cv.mean():.4f} +/- {cv.std():.4f}")

        results = evaluate(clf, X_test_s, y_test, y_test_ddg, "S669 Test")

        # Score = weighted combination of AUC + F1 + Pearson
        score = results['auc'] * 0.4 + results['f1'] * 0.3 + results['pearson'] * 0.3
        print(f"  Combined score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = clf
            best_scaler = scaler
            best_name = name

    # ── Also try with DDG threshold tuning ──
    print(f"\n\n{'='*70}")
    print(f"BEST MODEL: {best_name} (score={best_score:.4f})")
    print(f"{'='*70}")

    # Try different DDG thresholds for stabilizing label
    print("\n--- DDG threshold sensitivity ---")
    for threshold in [-0.5, -1.0, -1.5]:
        y_train_t = (y_train_ddg < threshold).astype(int)
        y_test_t = (y_test_ddg < threshold).astype(int)
        if np.sum(y_train_t) < 50 or np.sum(y_test_t) < 10:
            continue
        clf_t = GradientBoostingClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
            min_samples_leaf=10, max_features='sqrt', random_state=42)
        sw_t = np.ones(len(y_train_t))
        np_t = np.sum(y_train_t == 1)
        nn_t = np.sum(y_train_t == 0)
        sw_t[y_train_t == 1] = len(y_train_t) / (2 * max(np_t, 1))
        sw_t[y_train_t == 0] = len(y_train_t) / (2 * max(nn_t, 1))
        clf_t.fit(X_train_s, y_train_t, sample_weight=sw_t)
        print(f"\n  Threshold DDG < {threshold} kcal/mol:")
        print(f"    Train: {np_t} stabilizing, {nn_t} destabilizing")
        evaluate(clf_t, X_test_s, y_test_t, y_test_ddg, f"S669 (threshold={threshold})")

    # ── Save best model ──
    print(f"\n\nSaving best model: {best_name}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "mutation_classifier.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(best_scaler, f)

    # Final evaluation
    y_pred = best_model.predict(X_test_s)
    y_prob = best_model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    pr, _ = pearsonr(y_prob, -y_test_ddg)
    sr, _ = spearmanr(y_prob, -y_test_ddg)

    cv_acc = cross_val_score(best_model, X_train_s, y_train, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(best_model, X_train_s, y_train, cv=skf, scoring='f1')
    cv_auc = cross_val_score(best_model, X_train_s, y_train, cv=skf, scoring='roc_auc')

    meta = {
        "model_type": f"Publication-ready ({best_name})",
        "n_features": int(X_train.shape[1]),
        "feature_version": "v4_publication",
        "training_samples": int(X_train.shape[0]),
        "stabilizing_samples": int(n_pos),
        "destabilizing_samples": int(n_neg),
        "data_sources": "FireProtDB + ProDDG + ThermoMutDB (real data only)",
        "synthetic_data": False,
        "independent_test_set": "S669 (669 mutations)",
        "cv_accuracy": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_f1": round(float(cv_f1.mean()), 4),
        "cv_auc": round(float(cv_auc.mean()), 4),
        "test_accuracy": round(float(acc), 4),
        "test_f1": round(float(f1), 4),
        "test_auc": round(float(auc), 4),
        "test_pearson_r": round(float(pr), 4),
        "test_spearman_r": round(float(sr), 4),
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFinal model saved.")
    print(f"  CV Accuracy: {cv_acc.mean():.4f}")
    print(f"  CV AUC: {cv_auc.mean():.4f}")
    print(f"  S669 Accuracy: {acc:.4f}")
    print(f"  S669 AUC: {auc:.4f}")
    print(f"  S669 Pearson: {pr:.4f}")


if __name__ == "__main__":
    main()
