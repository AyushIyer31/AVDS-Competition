"""ESM-2 protein language model for embeddings and stability prediction."""

import torch
import numpy as np
from typing import Optional

# ESM-2 model (loaded lazily)
_model = None
_alphabet = None
_batch_converter = None

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _load_model():
    """Load ESM-2 model (650M parameter version)."""
    global _model, _alphabet, _batch_converter
    if _model is not None:
        return

    import esm

    _model, _alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    _batch_converter = _alphabet.get_batch_converter()
    _model.eval()

    # Use GPU if available
    if torch.cuda.is_available():
        _model = _model.cuda()


def get_embedding(sequence: str) -> np.ndarray:
    """Get per-residue ESM-2 embeddings for a protein sequence.

    Returns shape (seq_len, 1280) embedding matrix.
    """
    _load_model()

    data = [("protein", sequence)]
    _, _, batch_tokens = _batch_converter(data)

    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = _model(batch_tokens, repr_layers=[33], return_contacts=False)

    # Extract embeddings, remove BOS/EOS tokens
    embeddings = results["representations"][33][0, 1 : len(sequence) + 1].cpu().numpy()
    return embeddings


def get_sequence_embedding(sequence: str) -> np.ndarray:
    """Get a single mean-pooled embedding vector for the full sequence.

    Returns shape (1280,) vector.
    """
    per_residue = get_embedding(sequence)
    return per_residue.mean(axis=0)


def get_logits(sequence: str) -> np.ndarray:
    """Get per-position amino acid logits (mutation landscape).

    Returns shape (seq_len, 33) logit matrix.
    """
    _load_model()

    data = [("protein", sequence)]
    _, _, batch_tokens = _batch_converter(data)

    if torch.cuda.is_available():
        batch_tokens = batch_tokens.cuda()

    with torch.no_grad():
        results = _model(batch_tokens)

    logits = results["logits"][0, 1 : len(sequence) + 1].cpu().numpy()
    return logits


def predict_mutation_effect(sequence: str, position: int, mutant_aa: str) -> float:
    """Score a single point mutation using ESM-2 log-likelihood ratio.

    Positive scores suggest the mutation is favorable.
    """
    _load_model()

    logits = get_logits(sequence)
    # Get token indices for wild-type and mutant
    wt_aa = sequence[position]
    wt_idx = _alphabet.get_idx(wt_aa)
    mut_idx = _alphabet.get_idx(mutant_aa)

    # Log-likelihood ratio: positive means mutant is more likely
    log_probs = torch.nn.functional.log_softmax(torch.tensor(logits[position]), dim=0)
    score = (log_probs[mut_idx] - log_probs[wt_idx]).item()
    return score


def scan_beneficial_mutations(sequence: str, top_k: int = 20) -> list[dict]:
    """Scan all single-point mutations and return the top-k most beneficial.

    Uses ESM-2 masked marginal scoring to identify positions where
    mutations are predicted to improve the protein.
    """
    _load_model()

    logits = get_logits(sequence)
    log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=-1).numpy()

    mutations = []
    for pos in range(len(sequence)):
        wt_aa = sequence[pos]
        wt_idx = _alphabet.get_idx(wt_aa)
        wt_score = log_probs[pos, wt_idx]

        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            mut_idx = _alphabet.get_idx(mut_aa)
            mut_score = log_probs[pos, mut_idx]
            delta = float(mut_score - wt_score)

            if delta > 0:  # Only keep beneficial mutations
                mutations.append({
                    "position": pos,
                    "wild_type": wt_aa,
                    "mutant": mut_aa,
                    "score": delta,
                    "label": f"{wt_aa}{pos + 1}{mut_aa}",
                })

    # Sort by score descending, return top-k
    mutations.sort(key=lambda x: x["score"], reverse=True)
    return mutations[:top_k]
