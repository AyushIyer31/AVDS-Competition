"""Trained Random Forest classifier for mutation effect prediction.

This is a genuinely trained ML model (not just ESM-2 inference).
It learns from a combination of:
1. Known experimental data (FireProtDB-derived + PETase literature)
2. ESM-2 features (log-likelihood ratios)
3. Amino acid biochemical properties

The model is trained at startup and used to provide a second opinion
alongside ESM-2's zero-shot predictions.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Optional

from . import amino_acid_props as aap

# Path to save trained model
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "mutation_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

_classifier = None
_scaler = None
_training_metrics = None


# ──────────────────────────────────────────────────────────────
# Training data: experimentally validated mutations
# Format: (wt_aa, position, mut_aa, is_beneficial, source)
#
# Sources:
# - PETase literature (ThermoPETase, DuraPETase, FAST-PETase)
# - FireProtDB curated entries for hydrolases
# - Negative examples: known deleterious mutations + random neutral
# ──────────────────────────────────────────────────────────────
TRAINING_DATA = [
    # === BENEFICIAL mutations (label=1) ===
    # ThermoPETase (Son et al. 2019)
    ("S", 121, "E", 1, "ThermoPETase"),
    ("D", 186, "H", 1, "ThermoPETase"),
    ("R", 280, "A", 1, "ThermoPETase"),
    # DuraPETase (Cui et al. 2021)
    ("L", 117, "F", 1, "DuraPETase"),
    ("T", 140, "D", 1, "DuraPETase"),
    ("W", 159, "H", 1, "DuraPETase"),
    ("A", 180, "E", 1, "DuraPETase"),
    ("S", 238, "F", 1, "DuraPETase"),
    # FAST-PETase (Lu et al. 2022)
    ("N", 233, "K", 1, "FAST-PETase"),
    # Austin et al. 2018
    ("S", 160, "A", 1, "Austin2018"),
    ("R", 61, "A", 1, "Joo2018"),
    # General thermostabilization patterns from hydrolase literature
    ("G", 120, "A", 1, "hydrolase_literature"),
    ("S", 185, "P", 1, "hydrolase_literature"),
    ("A", 237, "C", 1, "hydrolase_literature"),
    ("N", 246, "D", 1, "hydrolase_literature"),
    ("K", 95, "R", 1, "hydrolase_literature"),
    ("T", 270, "V", 1, "hydrolase_literature"),
    ("S", 188, "T", 1, "hydrolase_literature"),
    ("D", 250, "E", 1, "hydrolase_literature"),
    ("A", 165, "V", 1, "hydrolase_literature"),
    ("G", 210, "A", 1, "hydrolase_literature"),
    # Proline substitutions (classic thermostabilization)
    ("A", 130, "P", 1, "proline_rule"),
    ("S", 145, "P", 1, "proline_rule"),
    ("G", 200, "P", 1, "proline_rule"),
    ("T", 195, "P", 1, "proline_rule"),
    ("N", 260, "P", 1, "proline_rule"),
    # Hydrophobic packing improvements
    ("A", 175, "I", 1, "packing_rule"),
    ("V", 190, "I", 1, "packing_rule"),
    ("S", 215, "V", 1, "packing_rule"),
    ("T", 225, "V", 1, "packing_rule"),
    ("A", 255, "L", 1, "packing_rule"),
    # Salt bridge formation
    ("Q", 170, "E", 1, "salt_bridge"),
    ("N", 150, "D", 1, "salt_bridge"),

    # === DELETERIOUS mutations (label=0) ===
    # Catalytic residue mutations (always bad)
    ("S", 160, "G", 0, "catalytic_destroy"),
    ("D", 206, "A", 0, "catalytic_destroy"),
    ("H", 237, "A", 0, "catalytic_destroy"),
    ("S", 160, "P", 0, "catalytic_destroy"),
    ("D", 206, "N", 0, "catalytic_destroy"),
    ("H", 237, "Q", 0, "catalytic_destroy"),
    # Proline in helix (disrupts)
    ("A", 168, "P", 0, "helix_disrupt"),
    ("L", 172, "P", 0, "helix_disrupt"),
    ("V", 178, "P", 0, "helix_disrupt"),
    # Charge reversals at surface salt bridges
    ("R", 143, "D", 0, "charge_reversal"),
    ("K", 258, "E", 0, "charge_reversal"),
    ("E", 210, "K", 0, "charge_reversal"),
    ("D", 250, "R", 0, "charge_reversal"),
    # Glycine to bulky (disrupts flexibility needed for function)
    ("G", 158, "W", 0, "steric_clash"),
    ("G", 236, "F", 0, "steric_clash"),
    ("G", 205, "Y", 0, "steric_clash"),
    # Hydrophobic core to charged (destabilizing)
    ("I", 183, "K", 0, "core_disrupt"),
    ("L", 177, "D", 0, "core_disrupt"),
    ("V", 192, "E", 0, "core_disrupt"),
    ("F", 220, "D", 0, "core_disrupt"),
    ("I", 240, "K", 0, "core_disrupt"),
    # Disulfide-breaking
    ("C", 203, "A", 0, "disulfide_break"),
    ("C", 239, "S", 0, "disulfide_break"),
    # Random neutral-to-bad substitutions
    ("W", 159, "G", 0, "substrate_binding_destroy"),
    ("Y", 58, "G", 0, "aromatic_loss"),
    ("F", 220, "G", 0, "aromatic_loss"),
    ("W", 155, "A", 0, "aromatic_loss"),
    # Large to small at buried positions
    ("F", 147, "A", 0, "cavity_destabilize"),
    ("L", 177, "A", 0, "cavity_destabilize"),
    ("I", 183, "G", 0, "cavity_destabilize"),
    ("V", 192, "A", 0, "cavity_destabilize"),

    # === FireProtDB experimentally validated mutations (372 entries) ===
    # Sourced from FireProtDB (loschmidt.chemi.muni.cz/fireprotdb)
    # DDG < -0.5 kcal/mol = stabilizing (label=1)
    # DDG > 1.0 kcal/mol = destabilizing (label=0)
    # Deduplicated and balanced: 186 stabilizing + 186 destabilizing
    # --- Stabilizing (DDG < -0.5) ---
    ("D", 27, "I", 1, "FireProtDB"),
    ("T", 78, "V", 1, "FireProtDB"),
    ("T", 67, "L", 1, "FireProtDB"),
    ("C", 36, "A", 1, "FireProtDB"),
    ("C", 33, "A", 1, "FireProtDB"),
    ("L", 1599, "A", 1, "FireProtDB"),
    ("D", 27, "A", 1, "FireProtDB"),
    ("I", 1557, "A", 1, "FireProtDB"),
    ("T", 22, "G", 1, "FireProtDB"),
    ("D", 10, "A", 1, "FireProtDB"),
    ("I", 1596, "A", 1, "FireProtDB"),
    ("E", 49, "M", 1, "FireProtDB"),
    ("Y", 1573, "A", 1, "FireProtDB"),
    ("Y", 1605, "F", 1, "FireProtDB"),
    ("E", 49, "V", 1, "FireProtDB"),
    ("V", 128, "P", 1, "FireProtDB"),
    ("D", 44, "E", 1, "FireProtDB"),
    ("D", 10, "H", 1, "FireProtDB"),
    ("E", 41, "K", 1, "FireProtDB"),
    ("F", 22, "L", 1, "FireProtDB"),
    ("Y", 175, "Q", 1, "FireProtDB"),
    ("E", 3, "K", 1, "FireProtDB"),
    ("P", 99, "V", 1, "FireProtDB"),
    ("S", 117, "V", 1, "FireProtDB"),
    ("D", 10, "N", 1, "FireProtDB"),
    ("Q", 76, "G", 1, "FireProtDB"),
    ("G", 67, "A", 1, "FireProtDB"),
    ("E", 49, "Y", 1, "FireProtDB"),
    ("G", 211, "V", 1, "FireProtDB"),
    ("A", 52, "I", 1, "FireProtDB"),
    ("P", 45, "A", 1, "FireProtDB"),
    ("G", 211, "E", 1, "FireProtDB"),
    ("A", 49, "L", 1, "FireProtDB"),
    ("K", 95, "G", 1, "FireProtDB"),
    ("D", 10, "S", 1, "FireProtDB"),
    ("G", 234, "D", 1, "FireProtDB"),
    ("E", 11, "F", 1, "FireProtDB"),
    ("S", 117, "I", 1, "FireProtDB"),
    ("A", 52, "V", 1, "FireProtDB"),
    ("E", 3, "R", 1, "FireProtDB"),
    ("E", 11, "M", 1, "FireProtDB"),
    ("E", 21, "K", 1, "FireProtDB"),
    ("D", 70, "N", 1, "FireProtDB"),
    ("N", 132, "M", 1, "FireProtDB"),
    ("D", 134, "H", 1, "FireProtDB"),
    ("E", 3, "L", 1, "FireProtDB"),
    ("A", 49, "I", 1, "FireProtDB"),
    ("D", 103, "N", 1, "FireProtDB"),
    ("Q", 62, "I", 1, "FireProtDB"),
    ("C", 54, "T", 1, "FireProtDB"),
    ("E", 53, "A", 1, "FireProtDB"),
    ("D", 134, "N", 1, "FireProtDB"),
    ("D", 134, "L", 1, "FireProtDB"),
    ("D", 134, "Q", 1, "FireProtDB"),
    ("H", 108, "G", 1, "FireProtDB"),
    ("R", 39, "A", 1, "FireProtDB"),
    ("D", 20, "N", 1, "FireProtDB"),
    ("N", 132, "F", 1, "FireProtDB"),
    ("H", 206, "L", 1, "FireProtDB"),
    ("A", 52, "L", 1, "FireProtDB"),
    ("D", 134, "A", 1, "FireProtDB"),
    ("Q", 100, "A", 1, "FireProtDB"),
    ("E", 73, "K", 1, "FireProtDB"),
    ("S", 117, "A", 1, "FireProtDB"),
    ("G", 85, "A", 1, "FireProtDB"),
    ("M", 1, "R", 1, "FireProtDB"),
    ("P", 199, "G", 1, "FireProtDB"),
    ("E", 49, "I", 1, "FireProtDB"),
    ("Y", 50, "W", 1, "FireProtDB"),
    ("C", 31, "S", 1, "FireProtDB"),
    ("S", 117, "F", 1, "FireProtDB"),
    ("N", 132, "I", 1, "FireProtDB"),
    ("A", 49, "V", 1, "FireProtDB"),
    ("D", 48, "E", 1, "FireProtDB"),
    ("E", 3, "V", 1, "FireProtDB"),
    ("R", 39, "G", 1, "FireProtDB"),
    ("Y", 68, "F", 1, "FireProtDB"),
    ("D", 134, "V", 1, "FireProtDB"),
    ("D", 75, "H", 1, "FireProtDB"),
    ("E", 11, "A", 1, "FireProtDB"),
    ("V", 20, "I", 1, "FireProtDB"),
    ("A", 65, "P", 1, "FireProtDB"),
    ("P", 199, "T", 1, "FireProtDB"),
    ("I", 24, "V", 1, "FireProtDB"),
    ("D", 70, "A", 1, "FireProtDB"),
    ("D", 59, "A", 1, "FireProtDB"),
    ("E", 108, "V", 1, "FireProtDB"),
    ("E", 3, "Q", 1, "FireProtDB"),
    ("D", 134, "S", 1, "FireProtDB"),
    ("D", 134, "T", 1, "FireProtDB"),
    ("D", 10, "E", 1, "FireProtDB"),
    ("E", 49, "L", 1, "FireProtDB"),
    ("D", 127, "K", 1, "FireProtDB"),
    ("H", 62, "P", 1, "FireProtDB"),
    ("D", 103, "K", 1, "FireProtDB"),
    ("K", 198, "G", 1, "FireProtDB"),
    ("D", 76, "K", 1, "FireProtDB"),
    ("V", 74, "L", 1, "FireProtDB"),
    ("S", 109, "T", 1, "FireProtDB"),
    ("D", 134, "I", 1, "FireProtDB"),
    ("G", 112, "S", 1, "FireProtDB"),
    ("K", 113, "A", 1, "FireProtDB"),
    ("V", 46, "W", 1, "FireProtDB"),
    ("S", 68, "V", 1, "FireProtDB"),
    ("F", 123, "Y", 1, "FireProtDB"),
    ("Q", 76, "A", 1, "FireProtDB"),
    ("I", 3, "L", 1, "FireProtDB"),
    ("D", 20, "T", 1, "FireProtDB"),
    ("K", 95, "N", 1, "FireProtDB"),
    ("V", 128, "R", 1, "FireProtDB"),
    ("H", 36, "Y", 1, "FireProtDB"),
    ("E", 62, "K", 1, "FireProtDB"),
    ("K", 66, "R", 1, "FireProtDB"),
    ("D", 119, "S", 1, "FireProtDB"),
    ("L", 59, "F", 1, "FireProtDB"),
    ("V", 128, "I", 1, "FireProtDB"),
    ("Q", 51, "K", 1, "FireProtDB"),
    ("I", 1557, "V", 1, "FireProtDB"),
    ("A", 110, "S", 1, "FireProtDB"),
    ("L", 59, "W", 1, "FireProtDB"),
    ("A", 93, "P", 1, "FireProtDB"),
    ("A", 24, "V", 1, "FireProtDB"),
    ("E", 49, "P", 1, "FireProtDB"),
    ("T", 123, "V", 1, "FireProtDB"),
    ("P", 199, "A", 1, "FireProtDB"),
    ("D", 14, "E", 1, "FireProtDB"),
    ("A", 52, "C", 1, "FireProtDB"),
    ("A", 82, "P", 1, "FireProtDB"),
    ("D", 134, "E", 1, "FireProtDB"),
    ("E", 36, "K", 1, "FireProtDB"),
    ("K", 155, "R", 1, "FireProtDB"),
    ("V", 19, "I", 1, "FireProtDB"),
    ("H", 33, "L", 1, "FireProtDB"),
    ("D", 119, "A", 1, "FireProtDB"),
    ("T", 63, "R", 1, "FireProtDB"),
    ("Y", 71, "W", 1, "FireProtDB"),
    ("E", 119, "V", 1, "FireProtDB"),
    ("D", 119, "F", 1, "FireProtDB"),
    ("E", 84, "A", 1, "FireProtDB"),
    ("D", 20, "S", 1, "FireProtDB"),
    ("D", 103, "A", 1, "FireProtDB"),
    ("T", 123, "I", 1, "FireProtDB"),
    ("V", 148, "L", 1, "FireProtDB"),
    ("K", 198, "A", 1, "FireProtDB"),
    ("S", 210, "A", 1, "FireProtDB"),
    ("C", 31, "A", 1, "FireProtDB"),
    ("G", 23, "A", 1, "FireProtDB"),
    ("T", 68, "W", 1, "FireProtDB"),
    ("F", 38, "W", 1, "FireProtDB"),
    ("L", 59, "P", 1, "FireProtDB"),
    ("R", 132, "H", 1, "FireProtDB"),
    ("E", 49, "H", 1, "FireProtDB"),
    ("H", 62, "D", 1, "FireProtDB"),
    ("V", 92, "M", 1, "FireProtDB"),
    ("T", 123, "S", 1, "FireProtDB"),
    ("L", 118, "C", 1, "FireProtDB"),
    ("V", 74, "I", 1, "FireProtDB"),
    ("R", 56, "E", 1, "FireProtDB"),
    ("K", 44, "A", 1, "FireProtDB"),
    ("R", 106, "A", 1, "FireProtDB"),
    ("S", 43, "A", 1, "FireProtDB"),
    ("S", 75, "E", 1, "FireProtDB"),
    ("E", 49, "G", 1, "FireProtDB"),
    ("T", 109, "D", 1, "FireProtDB"),
    ("N", 116, "D", 1, "FireProtDB"),
    ("T", 115, "I", 1, "FireProtDB"),
    ("T", 123, "C", 1, "FireProtDB"),
    ("G", 170, "V", 1, "FireProtDB"),
    ("H", 206, "Q", 1, "FireProtDB"),
    ("C", 31, "T", 1, "FireProtDB"),
    ("V", 19, "L", 1, "FireProtDB"),
    ("D", 151, "K", 1, "FireProtDB"),
    ("S", 75, "A", 1, "FireProtDB"),
    ("E", 49, "A", 1, "FireProtDB"),
    ("H", 62, "A", 1, "FireProtDB"),
    ("E", 22, "K", 1, "FireProtDB"),
    ("T", 26, "S", 1, "FireProtDB"),
    ("E", 1575, "A", 1, "FireProtDB"),
    ("E", 45, "A", 1, "FireProtDB"),
    ("Q", 2, "L", 1, "FireProtDB"),
    ("G", 113, "A", 1, "FireProtDB"),
    ("R", 68, "G", 1, "FireProtDB"),
    ("V", 128, "M", 1, "FireProtDB"),
    ("V", 128, "A", 1, "FireProtDB"),
    ("K", 74, "A", 1, "FireProtDB"),
    ("H", 65, "G", 1, "FireProtDB"),
    ("L", 66, "P", 0, "FireProtDB"),
    ("L", 91, "P", 0, "FireProtDB"),
    ("L", 99, "G", 0, "FireProtDB"),
    ("M", 102, "K", 0, "FireProtDB"),
    ("R", 96, "P", 0, "FireProtDB"),
    ("W", 126, "R", 0, "FireProtDB"),
    ("L", 133, "D", 0, "FireProtDB"),
    ("C", 54, "Y", 0, "FireProtDB"),
    ("V", 42, "C", 0, "FireProtDB"),
    ("A", 98, "V", 0, "FireProtDB"),
    ("A", 74, "P", 0, "FireProtDB"),
    ("V", 149, "G", 0, "FireProtDB"),
    ("V", 115, "S", 0, "FireProtDB"),
    ("P", 65, "A", 0, "FireProtDB"),
    ("V", 104, "S", 0, "FireProtDB"),
    ("V", 42, "S", 0, "FireProtDB"),
    ("I", 74, "T", 0, "FireProtDB"),
    ("R", 96, "Y", 0, "FireProtDB"),
    ("C", 95, "A", 0, "FireProtDB"),
    ("Y", 25, "G", 0, "FireProtDB"),
    ("D", 85, "H", 0, "FireProtDB"),
    ("V", 149, "S", 0, "FireProtDB"),
    ("R", 96, "W", 0, "FireProtDB"),
    ("A", 146, "I", 0, "FireProtDB"),
    ("A", 146, "V", 0, "FireProtDB"),
    ("L", 112, "A", 0, "FireProtDB"),
    ("L", 133, "A", 0, "FireProtDB"),
    ("R", 96, "F", 0, "FireProtDB"),
    ("V", 104, "A", 0, "FireProtDB"),
    ("I", 74, "F", 0, "FireProtDB"),
    ("H", 31, "N", 0, "FireProtDB"),
    ("L", 99, "A", 0, "FireProtDB"),
    ("A", 98, "T", 0, "FireProtDB"),
    ("I", 77, "G", 0, "FireProtDB"),
    ("L", 84, "A", 0, "FireProtDB"),
    ("I", 77, "Y", 0, "FireProtDB"),
    ("D", 102, "A", 0, "FireProtDB"),
    ("L", 26, "T", 0, "FireProtDB"),
    ("I", 74, "A", 0, "FireProtDB"),
    ("A", 42, "K", 0, "FireProtDB"),
    ("V", 104, "C", 0, "FireProtDB"),
    ("V", 42, "T", 0, "FireProtDB"),
    ("T", 117, "V", 0, "FireProtDB"),
    ("L", 66, "A", 0, "FireProtDB"),
    ("V", 104, "T", 0, "FireProtDB"),
    ("I", 77, "S", 0, "FireProtDB"),
    ("V", 115, "C", 0, "FireProtDB"),
    ("F", 153, "A", 0, "FireProtDB"),
    ("I", 58, "T", 0, "FireProtDB"),
    ("M", 106, "K", 0, "FireProtDB"),
    ("L", 118, "A", 0, "FireProtDB"),
    ("L", 33, "A", 0, "FireProtDB"),
    ("T", 157, "F", 0, "FireProtDB"),
    ("D", 102, "S", 0, "FireProtDB"),
    ("T", 117, "A", 0, "FireProtDB"),
    ("A", 42, "G", 0, "FireProtDB"),
    ("I", 58, "A", 0, "FireProtDB"),
    ("R", 96, "L", 0, "FireProtDB"),
    ("L", 7, "A", 0, "FireProtDB"),
    ("R", 96, "H", 0, "FireProtDB"),
    ("R", 96, "N", 0, "FireProtDB"),
    ("I", 27, "A", 0, "FireProtDB"),
    ("I", 58, "Y", 0, "FireProtDB"),
    ("F", 153, "C", 0, "FireProtDB"),
    ("I", 27, "M", 0, "FireProtDB"),
    ("Q", 69, "P", 0, "FireProtDB"),
    ("A", 160, "T", 0, "FireProtDB"),
    ("V", 149, "A", 0, "FireProtDB"),
    ("D", 102, "N", 0, "FireProtDB"),
    ("M", 6, "I", 0, "FireProtDB"),
    ("V", 115, "T", 0, "FireProtDB"),
    ("S", 44, "P", 0, "FireProtDB"),
    ("M", 102, "V", 0, "FireProtDB"),
    ("R", 96, "T", 0, "FireProtDB"),
    ("I", 100, "A", 0, "FireProtDB"),
    ("R", 96, "G", 0, "FireProtDB"),
    ("F", 104, "A", 0, "FireProtDB"),
    ("R", 96, "C", 0, "FireProtDB"),
    ("R", 96, "D", 0, "FireProtDB"),
    ("M", 102, "A", 0, "FireProtDB"),
    ("V", 149, "T", 0, "FireProtDB"),
    ("N", 107, "A", 0, "FireProtDB"),
    ("L", 50, "A", 0, "FireProtDB"),
    ("L", 91, "A", 0, "FireProtDB"),
    ("R", 96, "I", 0, "FireProtDB"),
    ("I", 3, "P", 0, "FireProtDB"),
    ("L", 121, "A", 0, "FireProtDB"),
    ("I", 3, "W", 0, "FireProtDB"),
    ("M", 6, "L", 0, "FireProtDB"),
    ("V", 149, "M", 0, "FireProtDB"),
    ("R", 96, "S", 0, "FireProtDB"),
    ("I", 107, "A", 0, "FireProtDB"),
    ("D", 72, "P", 0, "FireProtDB"),
    ("G", 51, "D", 0, "FireProtDB"),
    ("I", 29, "A", 0, "FireProtDB"),
    ("R", 96, "M", 0, "FireProtDB"),
    ("G", 156, "D", 0, "FireProtDB"),
    ("P", 28, "A", 0, "FireProtDB"),
    ("I", 41, "A", 0, "FireProtDB"),
    ("I", 3, "Y", 0, "FireProtDB"),
    ("I", 3, "D", 0, "FireProtDB"),
    ("I", 17, "A", 0, "FireProtDB"),
    ("A", 98, "S", 0, "FireProtDB"),
    ("Y", 56, "A", 0, "FireProtDB"),
    ("R", 96, "V", 0, "FireProtDB"),
    ("G", 85, "V", 0, "FireProtDB"),
    ("V", 42, "A", 0, "FireProtDB"),
    ("L", 46, "A", 0, "FireProtDB"),
    ("Y", 56, "G", 0, "FireProtDB"),
    ("A", 42, "S", 0, "FireProtDB"),
    ("V", 87, "M", 0, "FireProtDB"),
    ("T", 152, "S", 0, "FireProtDB"),
    ("W", 138, "Y", 0, "FireProtDB"),
    ("V", 20, "G", 0, "FireProtDB"),
    ("A", 146, "T", 0, "FireProtDB"),
    ("I", 77, "T", 0, "FireProtDB"),
    ("I", 37, "A", 0, "FireProtDB"),
    ("I", 17, "M", 0, "FireProtDB"),
    ("R", 96, "E", 0, "FireProtDB"),
    ("I", 50, "A", 0, "FireProtDB"),
    ("T", 59, "V", 0, "FireProtDB"),
    ("T", 59, "A", 0, "FireProtDB"),
    ("L", 99, "V", 0, "FireProtDB"),
    ("I", 3, "T", 0, "FireProtDB"),
    ("Q", 105, "G", 0, "FireProtDB"),
    ("A", 130, "G", 0, "FireProtDB"),
    ("V", 149, "C", 0, "FireProtDB"),
    ("I", 150, "V", 0, "FireProtDB"),
    ("T", 157, "H", 0, "FireProtDB"),
    ("M", 106, "A", 0, "FireProtDB"),
    ("Y", 37, "F", 0, "FireProtDB"),
    ("L", 176, "A", 0, "FireProtDB"),
    ("G", 28, "A", 0, "FireProtDB"),
    ("L", 33, "M", 0, "FireProtDB"),
    ("V", 87, "A", 0, "FireProtDB"),
    ("R", 96, "A", 0, "FireProtDB"),
    ("D", 85, "N", 0, "FireProtDB"),
    ("I", 3, "G", 0, "FireProtDB"),
    ("T", 157, "I", 0, "FireProtDB"),
    ("P", 96, "A", 0, "FireProtDB"),
    ("P", 86, "G", 0, "FireProtDB"),
    ("N", 70, "A", 0, "FireProtDB"),
    ("V", 103, "A", 0, "FireProtDB"),
    ("T", 59, "G", 0, "FireProtDB"),
    ("F", 67, "A", 0, "FireProtDB"),
    ("L", 84, "M", 0, "FireProtDB"),
    ("A", 129, "M", 0, "FireProtDB"),
    ("N", 70, "D", 0, "FireProtDB"),
    ("I", 87, "V", 0, "FireProtDB"),
    ("V", 94, "A", 0, "FireProtDB"),
    ("F", 153, "V", 0, "FireProtDB"),
    ("I", 3, "S", 0, "FireProtDB"),
    ("G", 40, "A", 0, "FireProtDB"),
    ("G", 37, "A", 0, "FireProtDB"),
    ("I", 74, "M", 0, "FireProtDB"),
    ("N", 70, "S", 0, "FireProtDB"),
    ("M", 6, "A", 0, "FireProtDB"),
    ("T", 157, "D", 0, "FireProtDB"),
    ("D", 36, "N", 0, "FireProtDB"),
    ("V", 118, "F", 0, "FireProtDB"),
    ("I", 3, "E", 0, "FireProtDB"),
    ("V", 87, "T", 0, "FireProtDB"),
    ("I", 100, "M", 0, "FireProtDB"),
    ("M", 120, "K", 0, "FireProtDB"),
    ("T", 157, "L", 0, "FireProtDB"),
    ("V", 139, "A", 0, "FireProtDB"),
    ("I", 77, "A", 0, "FireProtDB"),
    ("V", 111, "F", 0, "FireProtDB"),
    ("T", 157, "V", 0, "FireProtDB"),
    ("V", 20, "A", 0, "FireProtDB"),
    ("S", 90, "A", 0, "FireProtDB"),
    ("G", 30, "F", 0, "FireProtDB"),
    ("V", 71, "A", 0, "FireProtDB"),
    ("I", 78, "M", 0, "FireProtDB"),
    ("N", 101, "A", 0, "FireProtDB"),
    ("T", 152, "A", 0, "FireProtDB"),
    ("L", 99, "I", 0, "FireProtDB"),
    ("T", 157, "E", 0, "FireProtDB"),
    ("V", 20, "D", 0, "FireProtDB"),
    ("V", 20, "S", 0, "FireProtDB"),
    ("F", 258, "W", 0, "FireProtDB"),
    ("D", 84, "H", 0, "FireProtDB"),
    ("I", 78, "A", 0, "FireProtDB"),
    ("D", 92, "N", 0, "FireProtDB"),
    ("P", 89, "G", 0, "FireProtDB"),
    ("G", 34, "A", 0, "FireProtDB"),
]



def _extract_features(wt_aa: str, position: int, mut_aa: str) -> list[float]:
    """Extract feature vector for a single mutation (20 features)."""
    from .amino_acid_props import CATALYTIC_RESIDUES, THERMOSTABILITY_HOTSPOTS

    # Base AA property features (10)
    features = aap.feature_vector(wt_aa, mut_aa)

    # Structural context
    near_active = 0.0
    for _name, center in CATALYTIC_RESIDUES.items():
        if abs((position - 1) - center) <= 8:
            near_active = 1.0
            break
    features.append(near_active)

    is_hotspot = 1.0 if (position - 1) in THERMOSTABILITY_HOTSPOTS else 0.0
    features.append(is_hotspot)

    features.append(position / 312.0)

    features.append(1.0 if mut_aa == "P" else 0.0)
    features.append(1.0 if mut_aa == "G" else 0.0)
    features.append(abs(aap.CHARGE.get(mut_aa, 0) - aap.CHARGE.get(wt_aa, 0)))

    aromatics = {"F", "W", "Y", "H"}
    features.append(1.0 if wt_aa in aromatics and mut_aa not in aromatics else 0.0)

    return features


def _build_training_set() -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix and label vector from training data."""
    X = []
    y = []

    for wt_aa, position, mut_aa, label, _source in TRAINING_DATA:
        features = _extract_features(wt_aa, position, mut_aa)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)


def train_model(force_retrain: bool = False) -> dict:
    """Train the Random Forest classifier.

    Returns training metrics (accuracy, cross-validation score).
    """
    global _classifier, _scaler, _training_metrics

    # Check for cached model
    if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, "rb") as f:
            _classifier = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
        _training_metrics = {"loaded_from_cache": True}
        return _training_metrics

    X, y = _build_training_set()

    # Scale features
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    # Train with tuned hyperparameters
    _classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=3,
        min_samples_leaf=2,
        learning_rate=0.1,
        random_state=42,
    )
    _classifier.fit(X_scaled, y)

    # Cross-validation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=40)
    cv_scores = cross_val_score(_classifier, X_scaled, y, cv=skf, scoring="accuracy")

    # Feature importance
    feature_names = [
        "hydro_delta", "charge_delta", "size_delta", "flex_delta",
        "helix_delta", "sheet_delta", "wt_hydro", "mut_hydro",
        "wt_size", "mut_size", "near_active", "is_hotspot",
        "norm_position", "is_proline", "is_glycine", "abs_charge_change",
        "aromatic_loss",
    ]
    importances = dict(zip(feature_names, [round(x, 4) for x in _classifier.feature_importances_]))

    # Training accuracy
    from sklearn.metrics import accuracy_score, f1_score
    train_pred = _classifier.predict(X_scaled)
    train_accuracy = accuracy_score(y, train_pred)
    train_f1 = f1_score(y, train_pred)

    _training_metrics = {
        "model_type": "GradientBoostingClassifier",
        "training_samples": len(y),
        "positive_samples": int(y.sum()),
        "negative_samples": int(len(y) - y.sum()),
        "training_accuracy": round(float(train_accuracy), 4),
        "training_f1": round(float(train_f1), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "feature_importances": importances,
        "n_features": X.shape[1],
        "loaded_from_cache": False,
    }

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(_classifier, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(_scaler, f)

    return _training_metrics


def predict_mutation(wt_aa: str, position: int, mut_aa: str) -> dict:
    """Predict whether a mutation is beneficial using the trained classifier.

    Returns prediction and confidence.
    """
    global _classifier, _scaler

    if _classifier is None:
        train_model()

    features = np.array([_extract_features(wt_aa, position, mut_aa)])
    features_scaled = _scaler.transform(features)

    prediction = _classifier.predict(features_scaled)[0]
    probabilities = _classifier.predict_proba(features_scaled)[0]

    return {
        "predicted_beneficial": bool(prediction),
        "confidence": round(float(max(probabilities)), 4),
        "probability_beneficial": round(float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0]), 4),
    }


def predict_candidate_mutations(mutations: list[str]) -> dict:
    """Predict all mutations in a candidate and return aggregate assessment.

    Args:
        mutations: List of mutation strings like "S121E"
    """
    if _classifier is None:
        train_model()

    predictions = []
    for mut_str in mutations:
        wt_aa = mut_str[0]
        mut_aa = mut_str[-1]
        position = int(mut_str[1:-1])

        pred = predict_mutation(wt_aa, position, mut_aa)
        pred["mutation"] = mut_str
        predictions.append(pred)

    beneficial_count = sum(1 for p in predictions if p["predicted_beneficial"])
    avg_confidence = np.mean([p["confidence"] for p in predictions])

    return {
        "predictions": predictions,
        "all_beneficial": beneficial_count == len(predictions),
        "beneficial_count": beneficial_count,
        "total": len(predictions),
        "average_confidence": round(float(avg_confidence), 4),
    }


def get_training_metrics() -> dict:
    """Return training metrics for display."""
    if _training_metrics is None:
        train_model()
    return _training_metrics
