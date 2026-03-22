"""Amino acid biochemical properties for mutation explainability.

Each amino acid is characterized by physicochemical properties that
determine its role in protein structure and function.
"""

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Net charge at pH 7
CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0,
    "G": 0, "H": 0, "I": 0, "K": 1, "L": 0,
    "M": 0, "N": 0, "P": 0, "Q": 0, "R": 1,
    "S": 0, "T": 0, "V": 0, "W": 0, "Y": 0,
}

# Molecular weight of side chain (daltons, approximate)
SIZE = {
    "A": 71, "C": 103, "D": 115, "E": 129, "F": 147,
    "G": 57, "H": 137, "I": 113, "K": 128, "L": 113,
    "M": 131, "N": 114, "P": 97, "Q": 128, "R": 156,
    "S": 87, "T": 101, "V": 99, "W": 186, "Y": 163,
}

# Backbone flexibility (Bhatt et al. normalized B-factors)
FLEXIBILITY = {
    "A": 0.36, "C": 0.35, "D": 0.51, "E": 0.50, "F": 0.31,
    "G": 0.54, "H": 0.32, "I": 0.46, "K": 0.47, "L": 0.40,
    "M": 0.30, "N": 0.46, "P": 0.51, "Q": 0.49, "R": 0.53,
    "S": 0.51, "T": 0.44, "V": 0.39, "W": 0.31, "Y": 0.42,
}

# Helix propensity (Chou-Fasman, higher = more helical)
HELIX_PROPENSITY = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13,
    "G": 0.57, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
    "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11, "R": 0.98,
    "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}

# Beta-sheet propensity (Chou-Fasman)
SHEET_PROPENSITY = {
    "A": 0.83, "C": 1.19, "D": 0.54, "E": 0.37, "F": 1.38,
    "G": 0.75, "H": 0.87, "I": 1.60, "K": 0.74, "L": 1.30,
    "M": 1.05, "N": 0.89, "P": 0.55, "Q": 1.10, "R": 0.93,
    "S": 0.75, "T": 1.19, "V": 1.70, "W": 1.37, "Y": 1.47,
}

FULL_NAMES = {
    "A": "Alanine", "C": "Cysteine", "D": "Aspartate", "E": "Glutamate",
    "F": "Phenylalanine", "G": "Glycine", "H": "Histidine",
    "I": "Isoleucine", "K": "Lysine", "L": "Leucine", "M": "Methionine",
    "N": "Asparagine", "P": "Proline", "Q": "Glutamine", "R": "Arginine",
    "S": "Serine", "T": "Threonine", "V": "Valine", "W": "Tryptophan",
    "Y": "Tyrosine",
}

# Amino acid categories for reasoning
CATEGORIES = {
    "A": "small hydrophobic", "C": "sulfur-containing", "D": "negatively charged",
    "E": "negatively charged", "F": "aromatic hydrophobic", "G": "small flexible",
    "H": "aromatic polar", "I": "branched hydrophobic", "K": "positively charged",
    "L": "hydrophobic", "M": "sulfur-containing", "N": "polar amide",
    "P": "rigid cyclic", "Q": "polar amide", "R": "positively charged",
    "S": "small polar", "T": "small polar", "V": "branched hydrophobic",
    "W": "large aromatic", "Y": "aromatic polar",
}


# ──────────────────────────────────────────────────────────────
# PETase-specific structural constants
# ──────────────────────────────────────────────────────────────

# Key catalytic residues in IsPETase (0-indexed)
CATALYTIC_RESIDUES = {
    "S160": 159,   # Catalytic serine (nucleophile)
    "D206": 205,   # Catalytic aspartate
    "H237": 236,   # Catalytic histidine
    "W159": 158,   # Substrate binding
    "S238": 237,   # Thermostability hotspot
    "R280": 279,   # Surface loop
}

# Positions known to improve thermostability when mutated
THERMOSTABILITY_HOTSPOTS = [121, 158, 159, 186, 237, 238, 279, 280]


def property_deltas(wt: str, mut: str) -> dict:
    """Compute property changes between wild-type and mutant amino acid."""
    return {
        "hydrophobicity_delta": round(HYDROPHOBICITY.get(mut, 0) - HYDROPHOBICITY.get(wt, 0), 2),
        "charge_delta": CHARGE.get(mut, 0) - CHARGE.get(wt, 0),
        "size_delta": SIZE.get(mut, 0) - SIZE.get(wt, 0),
        "flexibility_delta": round(FLEXIBILITY.get(mut, 0) - FLEXIBILITY.get(wt, 0), 3),
        "helix_propensity_delta": round(HELIX_PROPENSITY.get(mut, 0) - HELIX_PROPENSITY.get(wt, 0), 2),
        "sheet_propensity_delta": round(SHEET_PROPENSITY.get(mut, 0) - SHEET_PROPENSITY.get(wt, 0), 2),
    }


def feature_vector(wt: str, mut: str) -> list[float]:
    """Return a flat feature vector for ML training: [hydro_delta, charge_delta, size_delta, flex_delta, helix_delta, sheet_delta, wt_hydro, mut_hydro, wt_size, mut_size]."""
    d = property_deltas(wt, mut)
    return [
        d["hydrophobicity_delta"],
        float(d["charge_delta"]),
        float(d["size_delta"]),
        d["flexibility_delta"],
        d["helix_propensity_delta"],
        d["sheet_propensity_delta"],
        HYDROPHOBICITY.get(wt, 0),
        HYDROPHOBICITY.get(mut, 0),
        float(SIZE.get(wt, 0)),
        float(SIZE.get(mut, 0)),
    ]
