"""Latent space optimization for PETase enzyme engineering.

Uses ESM-2 embeddings as a continuous latent space, then performs
gradient-based optimization to find sequences with improved
catalytic efficiency and thermal stability.
"""

import numpy as np
from typing import Optional
from . import esm_engine
from .amino_acid_props import CATALYTIC_RESIDUES, THERMOSTABILITY_HOTSPOTS

# Weights for the combined fitness score
STABILITY_WEIGHT = 0.5
ACTIVITY_WEIGHT = 0.5


def _score_candidate(sequence: str) -> tuple[float, float]:
    """Score a candidate for both stability and activity in ONE ESM-2 pass.

    Returns (stability_score, activity_score) both in 0-1 range.
    """
    import torch

    logits = esm_engine.get_logits(sequence)  # single ESM-2 forward pass
    log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=-1).numpy()

    # --- Stability: mean log-likelihood across full sequence ---
    total_ll = 0.0
    for i, aa in enumerate(sequence):
        aa_idx = esm_engine._alphabet.get_idx(aa)
        total_ll += log_probs[i, aa_idx]

    mean_ll = total_ll / len(sequence)
    stability_score = float(np.clip((mean_ll + 3.0) / 3.0, 0.0, 1.0))

    # --- Activity: mean log-likelihood around active site windows ---
    scored_ll = 0.0
    scored_count = 0

    for _name, center_pos in CATALYTIC_RESIDUES.items():
        for offset in range(-5, 6):
            pos = center_pos + offset
            if 0 <= pos < len(sequence):
                aa = sequence[pos]
                aa_idx = esm_engine._alphabet.get_idx(aa)
                scored_ll += log_probs[pos, aa_idx]
                scored_count += 1

    if scored_count == 0:
        activity_score = 0.5
    else:
        active_mean_ll = scored_ll / scored_count
        activity_score = float(np.clip((active_mean_ll + 3.0) / 3.0, 0.0, 1.0))

    return stability_score, activity_score


def _combine_beneficial_mutations(
    sequence: str,
    beneficial_mutations: list[dict],
    num_mutations: int = 3,
) -> str:
    """Combine top non-overlapping beneficial mutations into a candidate sequence."""
    seq_list = list(sequence)
    used_positions = set()
    applied = 0

    for mut in beneficial_mutations:
        pos = mut["position"]
        if pos in used_positions:
            continue
        # Don't mutate catalytic residues (too risky)
        if pos in CATALYTIC_RESIDUES.values():
            continue

        seq_list[pos] = mut["mutant"]
        used_positions.add(pos)
        applied += 1

        if applied >= num_mutations:
            break

    return "".join(seq_list)


def optimize(
    sequence: str,
    num_candidates: int = 10,
    optimization_steps: int = 50,
    target_temp: float = 60.0,
) -> dict:
    """Run latent space optimization to generate improved PETase candidates.

    Strategy:
    1. Compute ESM-2 embedding of the wild-type sequence
    2. Scan all single mutations for beneficial ones (ESM-2 log-likelihood)
    3. Combine top mutations into multi-mutant candidates
    4. Score each candidate for stability + activity
    5. Return ranked candidates

    Args:
        sequence: Wild-type amino acid sequence
        num_candidates: Number of candidate sequences to return
        optimization_steps: Number of mutation combinations to explore
        target_temp: Target operating temperature in Celsius

    Returns:
        Dict with candidates and latent space metadata
    """
    # Step 1: Scan for beneficial single mutations
    beneficial = esm_engine.scan_beneficial_mutations(sequence, top_k=optimization_steps)

    if not beneficial:
        return {
            "original_sequence": sequence,
            "candidates": [],
            "latent_space_summary": {"beneficial_mutations_found": 0},
        }

    # Step 3: Generate candidate sequences with different mutation combinations
    # First, deduplicate by position — keep only the best mutation per position
    best_per_position = {}
    for mut in beneficial:
        pos = mut["position"]
        if pos not in best_per_position or mut["score"] > best_per_position[pos]["score"]:
            best_per_position[pos] = mut
    unique_muts = sorted(best_per_position.values(), key=lambda x: x["score"], reverse=True)

    candidates = []
    seen_seqs = set()

    # Single mutants from each unique position
    for mut in unique_muts:
        seq_list = list(sequence)
        pos = mut["position"]
        if pos in CATALYTIC_RESIDUES.values():
            continue
        seq_list[pos] = mut["mutant"]
        candidate_seq = "".join(seq_list)
        if candidate_seq != sequence and candidate_seq not in seen_seqs:
            mutations = [f"{mut['wild_type']}{pos + 1}{mut['mutant']}"]
            candidates.append({"sequence": candidate_seq, "mutations": mutations, "num_mutations": 1})
            seen_seqs.add(candidate_seq)

    # Multi-mutant combos: combine top N non-overlapping mutations
    max_combo = min(6, len(unique_muts))
    for n_muts in range(2, max_combo + 1):
        # Sliding window with stride to get diverse combos
        for start in range(0, len(unique_muts) - n_muts + 1):
            subset = unique_muts[start : start + n_muts]
            candidate_seq = _combine_beneficial_mutations(sequence, subset, num_mutations=n_muts)
            if candidate_seq == sequence or candidate_seq in seen_seqs:
                continue
            mutations = []
            for i, (wt, mt) in enumerate(zip(sequence, candidate_seq)):
                if wt != mt:
                    mutations.append(f"{wt}{i + 1}{mt}")
            if mutations:
                candidates.append({"sequence": candidate_seq, "mutations": mutations, "num_mutations": len(mutations)})
                seen_seqs.add(candidate_seq)
            if len(candidates) >= optimization_steps:
                break
        if len(candidates) >= optimization_steps:
            break

    # Step 4: Pre-rank by mutation score sum, then only ESM-score the top candidates
    # This avoids running ESM-2 on all 30+ candidates (saves ~45 seconds)
    for cand in candidates:
        cand["mutation_score_sum"] = sum(
            m["score"] for m in beneficial
            if any(m["label"] == mut for mut in cand["mutations"])
        )
        # Hotspot bonus for pre-ranking
        hotspot_bonus = sum(
            0.5 for m in cand["mutations"]
            if int(m[1:-1]) - 1 in THERMOSTABILITY_HOTSPOTS
        )
        cand["mutation_score_sum"] += hotspot_bonus

    candidates.sort(key=lambda x: x["mutation_score_sum"], reverse=True)
    top_to_score = candidates[:num_candidates + 5]  # score a few extra for safety

    scored = []
    for cand in top_to_score:
        stability, activity = _score_candidate(cand["sequence"])  # single ESM-2 pass
        combined = STABILITY_WEIGHT * stability + ACTIVITY_WEIGHT * activity

        hotspot_bonus = sum(
            0.02 for m in cand["mutations"]
            if int(m[1:-1]) - 1 in THERMOSTABILITY_HOTSPOTS
        )
        combined += hotspot_bonus

        scored.append({
            "sequence": cand["sequence"],
            "mutations": cand["mutations"],
            "predicted_stability_score": round(stability, 6),
            "predicted_activity_score": round(activity, 6),
            "combined_score": round(combined, 6),
        })

    # Step 5: Rank and return top candidates
    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    top_candidates = scored[:num_candidates]

    for i, cand in enumerate(top_candidates):
        cand["rank"] = i + 1

    # Step 6: Add explainability, literature validation, and classifier predictions
    # Lazy imports to avoid circular dependency
    from . import explainability as _explain
    from . import literature_validation as _litval
    from . import trained_classifier as _clf

    # Build ESM score lookup from beneficial mutations
    esm_score_lookup = {m["label"]: m["score"] for m in beneficial}

    # Train classifier if needed (fast, happens once)
    _clf.train_model()

    for cand in top_candidates:
        # Explainability
        explanation = _explain.explain_candidate(
            cand["mutations"], esm_scores=esm_score_lookup
        )
        cand["explanations"] = explanation["mutation_explanations"]
        cand["overall_strategy"] = explanation["overall_strategy"]

        # Literature validation
        validation = _litval.validate_mutations(cand["mutations"])
        cand["literature_validation"] = {
            "exact_matches": validation["exact_matches"],
            "position_matches": validation["position_matches"],
            "novel_predictions": validation["novel_predictions"],
            "variant_overlaps": validation["variant_overlaps"],
            "validation_score": validation["validation_score"],
            "summary": validation["summary"],
        }

        # Trained classifier prediction
        classifier_result = _clf.predict_candidate_mutations(cand["mutations"])
        cand["classifier_prediction"] = {
            "all_beneficial": classifier_result["all_beneficial"],
            "beneficial_count": classifier_result["beneficial_count"],
            "total": classifier_result["total"],
            "average_confidence": classifier_result["average_confidence"],
            "per_mutation": classifier_result["predictions"],
        }

    # Latent space summary — use stability/activity scores as 2D coordinates
    coords_2d = [[0.0, 0.0]]  # wild-type at origin
    for cand in top_candidates[:5]:
        x = (cand["predicted_stability_score"] - 0.9) * 20
        y = (cand["predicted_activity_score"] - 0.9) * 20
        coords_2d.append([round(x, 4), round(y, 4)])

    # Training metrics for display
    training_info = _clf.get_training_metrics()

    return {
        "original_sequence": sequence,
        "candidates": top_candidates,
        "latent_space_summary": {
            "beneficial_mutations_found": len(beneficial),
            "candidates_explored": len(candidates),
            "top_mutations": [m["label"] for m in beneficial[:10]],
            "latent_coordinates": coords_2d,
            "labels": ["wild_type"] + [f"candidate_{i+1}" for i in range(len(coords_2d) - 1)],
        },
        "classifier_info": {
            "model_type": training_info.get("model_type", "GradientBoostingClassifier"),
            "training_samples": training_info.get("training_samples", 0),
            "cv_accuracy": training_info.get("cv_accuracy_mean", 0),
            "feature_importances": training_info.get("feature_importances", {}),
        },
    }
