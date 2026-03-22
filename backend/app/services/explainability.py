"""Mutation explainability engine for PETase variants.

Generates human-readable biochemical explanations for why each
mutation is predicted to be beneficial or harmful.
"""

from . import amino_acid_props as aap
from .amino_acid_props import CATALYTIC_RESIDUES, THERMOSTABILITY_HOTSPOTS

# Distance thresholds for structural context
NEAR_ACTIVE_SITE_WINDOW = 8  # positions within this range of catalytic residues


def _is_near_active_site(position: int) -> bool:
    """Check if a position is near the catalytic site."""
    for _name, center in CATALYTIC_RESIDUES.items():
        if abs(position - center) <= NEAR_ACTIVE_SITE_WINDOW:
            return True
    return False


def _is_thermostability_hotspot(position: int) -> bool:
    return position in THERMOSTABILITY_HOTSPOTS


def explain_mutation(wt_aa: str, mut_aa: str, position: int, esm_score: float = 0.0) -> dict:
    """Generate a structured explanation for a single point mutation.

    Args:
        wt_aa: Wild-type amino acid (single letter)
        mut_aa: Mutant amino acid (single letter)
        position: 0-indexed position in the sequence
        esm_score: ESM-2 log-likelihood ratio score

    Returns:
        Dict with explanation fields
    """
    deltas = aap.property_deltas(wt_aa, mut_aa)
    wt_name = aap.FULL_NAMES.get(wt_aa, wt_aa)
    mut_name = aap.FULL_NAMES.get(mut_aa, mut_aa)
    wt_cat = aap.CATEGORIES.get(wt_aa, "unknown")
    mut_cat = aap.CATEGORIES.get(mut_aa, "unknown")

    reasons = []
    effects = []
    risk_level = "low"

    # --- Structural context ---
    near_active = _is_near_active_site(position)
    is_hotspot = _is_thermostability_hotspot(position)

    if near_active:
        reasons.append(f"Position {position + 1} is near the catalytic site — changes here directly affect PET binding and degradation.")
        risk_level = "medium"

    if is_hotspot:
        reasons.append(f"Position {position + 1} is a known thermostability hotspot — mutations here can significantly improve heat resistance.")

    # --- Hydrophobicity change ---
    hd = deltas["hydrophobicity_delta"]
    if abs(hd) > 3.0:
        if hd > 0:
            reasons.append(f"Large hydrophobicity increase (+{hd}) — the core becomes more tightly packed, which typically improves stability.")
            effects.append("stability_increase")
        else:
            reasons.append(f"Large hydrophobicity decrease ({hd}) — introduces polarity, can improve solubility but may reduce core packing.")
            effects.append("solubility_increase")
    elif abs(hd) > 1.0:
        direction = "more hydrophobic" if hd > 0 else "more polar"
        reasons.append(f"Moderate hydrophobicity shift ({'+' if hd > 0 else ''}{hd}) — residue becomes {direction}.")

    # --- Charge change ---
    cd = deltas["charge_delta"]
    if cd != 0:
        if cd > 0:
            reasons.append(f"Introduces positive charge — can form new salt bridges or improve surface interactions.")
            effects.append("charge_gain")
        else:
            reasons.append(f"Removes positive charge or adds negative charge — alters electrostatic network.")
            effects.append("charge_change")

    # --- Size change ---
    sd = deltas["size_delta"]
    if abs(sd) > 40:
        if sd > 0:
            reasons.append(f"Significantly larger side chain (+{sd} Da) — may fill cavities and improve packing density.")
        else:
            reasons.append(f"Significantly smaller side chain ({sd} Da) — creates space, reduces steric clashes.")
            effects.append("cavity_creation")

    # --- Flexibility change ---
    fd = deltas["flexibility_delta"]
    if abs(fd) > 0.1:
        if fd < 0:
            reasons.append(f"Reduces backbone flexibility — rigidifies the local structure, often improves thermostability.")
            effects.append("rigidification")
        else:
            reasons.append(f"Increases backbone flexibility — may improve conformational sampling for catalysis.")
            effects.append("flexibility_increase")

    # --- Proline substitution (special case) ---
    if mut_aa == "P" and wt_aa != "P":
        reasons.append("Proline introduction constrains the backbone — a classic strategy for thermostabilization.")
        effects.append("proline_rigidification")

    # --- Disulfide potential ---
    if mut_aa == "C" and wt_aa != "C":
        reasons.append("Cysteine introduction could enable a new disulfide bond if a partner Cys is nearby.")
        effects.append("disulfide_potential")
        risk_level = "medium"

    # --- ESM-2 evolutionary signal ---
    if esm_score > 0.5:
        reasons.append(f"ESM-2 strongly favors this mutation (score: {esm_score:.3f}) — the evolutionary language model predicts this residue fits better in this structural context.")
    elif esm_score > 0:
        reasons.append(f"ESM-2 moderately favors this mutation (score: {esm_score:.3f}) — predicted to be evolutionarily compatible.")

    # --- Category transition ---
    if wt_cat != mut_cat:
        reasons.append(f"Amino acid type changes from {wt_cat} ({wt_name}) to {mut_cat} ({mut_name}).")

    # Fallback
    if not reasons:
        reasons.append(f"Conservative substitution: {wt_name} to {mut_name} with similar properties.")

    # Build summary sentence
    summary = _build_summary(wt_name, mut_name, position, effects, near_active, is_hotspot)

    return {
        "mutation": f"{wt_aa}{position + 1}{mut_aa}",
        "from_aa": wt_name,
        "to_aa": mut_name,
        "from_category": wt_cat,
        "to_category": mut_cat,
        "position": position + 1,
        "summary": summary,
        "reasons": reasons,
        "effects": effects,
        "property_changes": deltas,
        "near_active_site": near_active,
        "thermostability_hotspot": is_hotspot,
        "risk_level": risk_level,
        "esm_score": round(esm_score, 4),
    }


def _build_summary(wt_name: str, mut_name: str, position: int, effects: list, near_active: bool, is_hotspot: bool) -> str:
    """Build a one-line human-readable summary."""
    parts = []

    if "stability_increase" in effects or "rigidification" in effects or "proline_rigidification" in effects:
        parts.append("improves thermostability")
    if "solubility_increase" in effects:
        parts.append("enhances solubility")
    if "charge_gain" in effects or "charge_change" in effects:
        parts.append("modifies electrostatic interactions")
    if "cavity_creation" in effects:
        parts.append("reduces steric clashes")
    if "flexibility_increase" in effects:
        parts.append("increases catalytic flexibility")
    if "disulfide_potential" in effects:
        parts.append("potential disulfide bond")

    if not parts:
        parts.append("conservative change with minimal structural impact")

    location = ""
    if is_hotspot:
        location = " at a thermostability hotspot"
    elif near_active:
        location = " near the active site"

    return f"{wt_name} to {mut_name} at position {position + 1}{location} — {', '.join(parts)}."


def explain_candidate(mutations: list[str], esm_scores: dict[str, float] | None = None) -> dict:
    """Explain all mutations in a candidate sequence.

    Args:
        mutations: List of mutation strings like "A65G"
        esm_scores: Optional dict mapping mutation label to ESM-2 score

    Returns:
        Dict with per-mutation explanations and overall assessment
    """
    esm_scores = esm_scores or {}
    explanations = []

    for mut_str in mutations:
        wt_aa = mut_str[0]
        mut_aa = mut_str[-1]
        position = int(mut_str[1:-1]) - 1  # convert to 0-indexed

        score = esm_scores.get(mut_str, 0.0)
        explanation = explain_mutation(wt_aa, mut_aa, position, esm_score=score)
        explanations.append(explanation)

    # Overall assessment
    stability_effects = sum(1 for e in explanations if any(
        x in e["effects"] for x in ["stability_increase", "rigidification", "proline_rigidification"]))
    activity_effects = sum(1 for e in explanations if any(
        x in e["effects"] for x in ["flexibility_increase", "charge_gain"]))
    hotspot_count = sum(1 for e in explanations if e["thermostability_hotspot"])

    if stability_effects > activity_effects:
        strategy = "thermostability-focused"
    elif activity_effects > stability_effects:
        strategy = "activity-focused"
    else:
        strategy = "balanced"

    return {
        "mutation_explanations": explanations,
        "overall_strategy": strategy,
        "stability_mutations": stability_effects,
        "activity_mutations": activity_effects,
        "hotspot_mutations": hotspot_count,
        "total_mutations": len(mutations),
    }
