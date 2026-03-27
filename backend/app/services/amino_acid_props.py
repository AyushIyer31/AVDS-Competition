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


# Volume (Å³) — more precise than molecular weight for steric effects
VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

# Isoelectric point (pI)
PI = {
    "A": 6.00, "C": 5.07, "D": 2.77, "E": 3.22, "F": 5.48,
    "G": 5.97, "H": 7.59, "I": 6.02, "K": 9.74, "L": 5.98,
    "M": 5.74, "N": 5.41, "P": 6.30, "Q": 5.65, "R": 10.76,
    "S": 5.68, "T": 5.60, "V": 5.96, "W": 5.89, "Y": 5.66,
}

# Burial propensity (higher = more likely buried in core)
BURIAL = {
    "A": 0.7, "C": 0.8, "D": 0.2, "E": 0.2, "F": 0.9,
    "G": 0.5, "H": 0.4, "I": 1.0, "K": 0.1, "L": 0.9,
    "M": 0.8, "N": 0.3, "P": 0.3, "Q": 0.3, "R": 0.1,
    "S": 0.4, "T": 0.5, "V": 0.9, "W": 0.7, "Y": 0.5,
}

# BLOSUM62 substitution likelihood (self-scores for conservation)
BLOSUM62_SELF = {
    "A": 4, "C": 9, "D": 6, "E": 5, "F": 6,
    "G": 6, "H": 8, "I": 4, "K": 5, "L": 4,
    "M": 5, "N": 6, "P": 7, "Q": 5, "R": 5,
    "S": 4, "T": 5, "V": 4, "W": 11, "Y": 7,
}

# BLOSUM62 pairwise scores (common pairs)
_BLOSUM62 = {
    ("A","A"):4,("A","R"):-1,("A","N"):-2,("A","D"):-2,("A","C"):0,("A","Q"):-1,("A","E"):-1,("A","G"):0,("A","H"):-2,("A","I"):-1,("A","L"):-1,("A","K"):-1,("A","M"):-1,("A","F"):-2,("A","P"):-1,("A","S"):1,("A","T"):0,("A","W"):-3,("A","Y"):-2,("A","V"):0,
    ("R","R"):5,("R","N"):0,("R","D"):-2,("R","C"):-3,("R","Q"):1,("R","E"):0,("R","G"):-2,("R","H"):0,("R","I"):-3,("R","L"):-2,("R","K"):2,("R","M"):-1,("R","F"):-3,("R","P"):-2,("R","S"):-1,("R","T"):-1,("R","W"):-3,("R","Y"):-2,("R","V"):-3,
    ("N","N"):6,("N","D"):1,("N","C"):-3,("N","Q"):0,("N","E"):0,("N","G"):-0,("N","H"):1,("N","I"):-3,("N","L"):-3,("N","K"):0,("N","M"):-2,("N","F"):-3,("N","P"):-2,("N","S"):1,("N","T"):0,("N","W"):-4,("N","Y"):-2,("N","V"):-3,
    ("D","D"):6,("D","C"):-3,("D","Q"):0,("D","E"):2,("D","G"):-1,("D","H"):-1,("D","I"):-3,("D","L"):-4,("D","K"):-1,("D","M"):-3,("D","F"):-3,("D","P"):-1,("D","S"):0,("D","T"):-1,("D","W"):-4,("D","Y"):-3,("D","V"):-3,
    ("C","C"):9,("C","Q"):-3,("C","E"):-4,("C","G"):-3,("C","H"):-3,("C","I"):-1,("C","L"):-1,("C","K"):-3,("C","M"):-1,("C","F"):-2,("C","P"):-3,("C","S"):-1,("C","T"):-1,("C","W"):-2,("C","Y"):-2,("C","V"):-1,
    ("Q","Q"):5,("Q","E"):2,("Q","G"):-2,("Q","H"):0,("Q","I"):-3,("Q","L"):-2,("Q","K"):1,("Q","M"):0,("Q","F"):-3,("Q","P"):-1,("Q","S"):0,("Q","T"):-1,("Q","W"):-2,("Q","Y"):-1,("Q","V"):-2,
    ("E","E"):5,("E","G"):-2,("E","H"):0,("E","I"):-3,("E","L"):-3,("E","K"):1,("E","M"):-2,("E","F"):-3,("E","P"):-1,("E","S"):0,("E","T"):-1,("E","W"):-3,("E","Y"):-2,("E","V"):-2,
    ("G","G"):6,("G","H"):-2,("G","I"):-4,("G","L"):-4,("G","K"):-2,("G","M"):-3,("G","F"):-3,("G","P"):-2,("G","S"):0,("G","T"):-2,("G","W"):-2,("G","Y"):-3,("G","V"):-3,
    ("H","H"):8,("H","I"):-3,("H","L"):-3,("H","K"):-1,("H","M"):-2,("H","F"):-1,("H","P"):-2,("H","S"):-1,("H","T"):-2,("H","W"):-2,("H","Y"):2,("H","V"):-3,
    ("I","I"):4,("I","L"):2,("I","K"):-3,("I","M"):1,("I","F"):0,("I","P"):-3,("I","S"):-2,("I","T"):-1,("I","W"):-3,("I","Y"):-1,("I","V"):3,
    ("L","L"):4,("L","K"):-2,("L","M"):2,("L","F"):0,("L","P"):-3,("L","S"):-2,("L","T"):-1,("L","W"):-2,("L","Y"):-1,("L","V"):1,
    ("K","K"):5,("K","M"):-1,("K","F"):-3,("K","P"):-1,("K","S"):0,("K","T"):-1,("K","W"):-3,("K","Y"):-2,("K","V"):-2,
    ("M","M"):5,("M","F"):0,("M","P"):-2,("M","S"):-1,("M","T"):-1,("M","W"):-1,("M","Y"):-1,("M","V"):1,
    ("F","F"):6,("F","P"):-4,("F","S"):-2,("F","T"):-2,("F","W"):1,("F","Y"):3,("F","V"):-1,
    ("P","P"):7,("P","S"):-1,("P","T"):-1,("P","W"):-4,("P","Y"):-3,("P","V"):-2,
    ("S","S"):4,("S","T"):1,("S","W"):-3,("S","Y"):-2,("S","V"):-2,
    ("T","T"):5,("T","W"):-2,("T","Y"):-2,("T","V"):0,
    ("V","V"):4,("V","W"):-3,("V","Y"):-1,
    ("W","W"):11,("W","Y"):2,
    ("Y","Y"):7,
}


def blosum62_score(wt: str, mut: str) -> float:
    """Get BLOSUM62 substitution score for a mutation."""
    key = (wt, mut) if (wt, mut) in _BLOSUM62 else (mut, wt)
    return float(_BLOSUM62.get(key, -1))


def feature_vector(wt: str, mut: str) -> list[float]:
    """Return a flat feature vector for ML training."""
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


def feature_vector_v2(wt: str, mut: str) -> list[float]:
    """Extended feature vector with 27 features for improved accuracy."""
    d = property_deltas(wt, mut)
    aromatics = {"F", "W", "Y", "H"}
    polar = {"S", "T", "N", "Q", "C"}
    charged = {"D", "E", "K", "R"}
    hydrophobic = {"A", "V", "I", "L", "M", "F", "W", "P"}

    return [
        # Property deltas (6)
        d["hydrophobicity_delta"],
        float(d["charge_delta"]),
        float(d["size_delta"]),
        d["flexibility_delta"],
        d["helix_propensity_delta"],
        d["sheet_propensity_delta"],
        # Absolute properties (4)
        HYDROPHOBICITY.get(wt, 0),
        HYDROPHOBICITY.get(mut, 0),
        float(SIZE.get(wt, 0)),
        float(SIZE.get(mut, 0)),
        # Volume delta (1) — more precise than size for steric
        VOLUME.get(mut, 0) - VOLUME.get(wt, 0),
        # Burial propensity change (1)
        BURIAL.get(mut, 0) - BURIAL.get(wt, 0),
        # BLOSUM62 substitution score (1) — evolutionary likelihood
        blosum62_score(wt, mut),
        # Conservation self-score of WT (1) — high = more conserved = riskier to mutate
        float(BLOSUM62_SELF.get(wt, 4)),
        # Category transitions (5)
        1.0 if wt in aromatics and mut not in aromatics else 0.0,  # aromatic loss
        1.0 if wt not in aromatics and mut in aromatics else 0.0,  # aromatic gain
        1.0 if wt in charged and mut not in charged else 0.0,  # charge loss
        1.0 if wt not in charged and mut in charged else 0.0,  # charge gain
        1.0 if wt in hydrophobic and mut in polar else 0.0,  # hydrophobic to polar
        # Special residue flags (4)
        1.0 if mut == "P" else 0.0,  # to proline (rigidifies backbone)
        1.0 if wt == "P" else 0.0,  # from proline (removes rigidity)
        1.0 if mut == "G" else 0.0,  # to glycine (adds flexibility)
        1.0 if wt == "G" else 0.0,  # from glycine (removes flexibility)
        # Interaction features (2)
        abs(d["hydrophobicity_delta"]) * abs(d["size_delta"]),  # hydro × size interaction
        abs(float(d["charge_delta"])) * d["flexibility_delta"],  # charge × flex interaction
    ]
