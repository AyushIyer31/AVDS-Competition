"""Generate PSSM-based conservation scores for all training proteins.

Runs PSI-BLAST against SwissProt to build PSSMs, then extracts
per-position conservation scores for use as ML features.

Output: conservation_cache.pkl — dict mapping (protein_id, sequence_hash) to
        per-position conservation arrays.
"""

import os
import sys
import json
import hashlib
import pickle
import subprocess
import tempfile
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PSI-BLAST can't handle spaces in paths, so use symlink
_blastdb_real = os.path.join(BASE_DIR, "blastdb")
BLASTDB_LINK = "/tmp/blastdb_petlab"
if not os.path.exists(BLASTDB_LINK):
    os.symlink(_blastdb_real, BLASTDB_LINK)
BLASTDB = os.path.join(BLASTDB_LINK, "swissprot")
CACHE_PATH = os.path.join(BASE_DIR, "conservation_cache.pkl")
# Use space-free path for PSSM output too
PSSM_DIR = "/tmp/pssm_cache_petlab"

FIREPROT_PATH = os.path.join(BASE_DIR, "fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")
PRODDG_PATH = os.path.join(BASE_DIR, "proddg_s2648.csv")
S669_PATH = os.path.join(BASE_DIR, "s669_full.tsv")
THERMOMUTDB_PATH = os.path.join(BASE_DIR, "thermomutdb.json")

AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")


def seq_hash(sequence):
    return hashlib.md5(sequence.encode()).hexdigest()[:12]


def collect_unique_proteins():
    """Collect all unique (protein_id, sequence) pairs from all datasets."""
    proteins = {}  # protein_id -> sequence

    # FireProtDB
    print("Loading FireProtDB sequences...")
    df = pd.read_csv(FIREPROT_PATH)
    for _, row in df.iterrows():
        pdb = str(row.get('pdb_id', '')).split('|')[0]
        seq = str(row.get('sequence', ''))
        if pdb and seq and len(seq) > 10 and pdb != 'nan':
            if pdb not in proteins or len(seq) > len(proteins[pdb]):
                proteins[pdb] = seq

    # ProDDG
    print("Loading ProDDG sequences...")
    df2 = pd.read_csv(PRODDG_PATH, sep='\t')
    for _, row in df2.iterrows():
        pdb = str(row.get('pdb', ''))
        seq = str(row.get('wt_sequence', ''))
        if pdb and seq and len(seq) > 10 and pdb != 'nan':
            if pdb not in proteins or len(seq) > len(proteins[pdb]):
                proteins[pdb] = seq

    # S669
    print("Loading S669 sequences...")
    df3 = pd.read_csv(S669_PATH, sep='\t')
    for _, row in df3.iterrows():
        pdb = str(row.get('pdb', ''))
        seq = str(row.get('wt_sequence', ''))
        if pdb and seq and len(seq) > 10 and pdb != 'nan':
            if pdb not in proteins or len(seq) > len(proteins[pdb]):
                proteins[pdb] = seq

    # ThermoMutDB — no sequences available, skip
    print(f"Collected {len(proteins)} unique proteins with sequences")
    return proteins


def run_psiblast(protein_id, sequence, num_iterations=3, evalue=0.001):
    """Run PSI-BLAST and return the PSSM matrix."""
    pssm_file = os.path.join(PSSM_DIR, f"{protein_id}.pssm")

    # Check cache
    if os.path.exists(pssm_file):
        return parse_pssm(pssm_file, len(sequence))

    # Write query FASTA
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        f.write(f">{protein_id}\n{sequence}\n")
        query_file = f.name

    try:
        cmd = [
            'psiblast',
            '-query', query_file,
            '-db', BLASTDB,
            '-num_iterations', str(num_iterations),
            '-evalue', str(evalue),
            '-out_ascii_pssm', pssm_file,
            '-num_threads', '4',
            '-num_alignments', '0',
        ]
        # Redirect stdout/stderr to devnull to avoid buffer issues
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(cmd, stdout=devnull, stderr=devnull, timeout=300)

        if os.path.exists(pssm_file) and os.path.getsize(pssm_file) > 100:
            parsed = parse_pssm(pssm_file, len(sequence))
            if parsed is not None:
                return parsed
            else:
                print(f"    WARNING: PSSM exists for {protein_id} but parsing failed")
        else:
            print(f"    WARNING: No PSSM generated for {protein_id}")

        return None

    except subprocess.TimeoutExpired:
        print(f"    WARNING: PSI-BLAST timed out for {protein_id}")
        return None
    except Exception as e:
        print(f"    WARNING: Error for {protein_id}: {e}")
        return None
    finally:
        if os.path.exists(query_file):
            os.unlink(query_file)


def parse_pssm(pssm_file, seq_len):
    """Parse PSI-BLAST ASCII PSSM file into a numpy array.

    Returns:
        pssm: (seq_len, 20) array of position-specific scores
        info_content: (seq_len,) array of information content per position
    """
    pssm_scores = []
    info_content = []

    with open(pssm_file) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # PSSM data lines start with a position number
        if len(parts) >= 44 and parts[0].isdigit():
            # Columns: pos, aa, then 20 log-odds scores, then 20 frequencies, then info, weight
            try:
                scores = [int(parts[i]) for i in range(2, 22)]  # 20 log-odds scores
                pssm_scores.append(scores)
                # Information content is the second-to-last column
                if len(parts) >= 44:
                    info_content.append(float(parts[42]))
                else:
                    info_content.append(0.0)
            except (ValueError, IndexError):
                continue

    if not pssm_scores:
        return None

    pssm = np.array(pssm_scores)
    info = np.array(info_content)

    return {'pssm': pssm, 'info_content': info}


def compute_conservation_features(pssm_data, position, wt_aa, mut_aa):
    """Extract conservation features from PSSM for a specific mutation.

    Returns 6 features:
        1. PSSM score for wild-type AA at this position (how expected is the WT)
        2. PSSM score for mutant AA at this position (how expected is the MUT)
        3. Delta PSSM (mut - wt): negative = mutation to less-favored AA
        4. Information content at position (higher = more conserved)
        5. Position conservation rank (0-1, how conserved relative to whole protein)
        6. Wild-type PSSM rank among all 20 AAs at this position
    """
    if pssm_data is None:
        return [0.0] * 6

    pssm = pssm_data['pssm']
    info = pssm_data['info_content']

    # Position is 1-indexed
    idx = position - 1
    if idx < 0 or idx >= len(pssm):
        return [0.0] * 6

    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}
    wt_idx = aa_to_idx.get(wt_aa)
    mut_idx = aa_to_idx.get(mut_aa)

    if wt_idx is None or mut_idx is None:
        return [0.0] * 6

    row = pssm[idx]

    # Feature 1: PSSM score for WT
    pssm_wt = float(row[wt_idx])

    # Feature 2: PSSM score for MUT
    pssm_mut = float(row[mut_idx])

    # Feature 3: Delta PSSM
    delta_pssm = pssm_mut - pssm_wt

    # Feature 4: Information content at this position
    info_at_pos = float(info[idx]) if idx < len(info) else 0.0

    # Feature 5: Relative conservation (rank of info content)
    if len(info) > 1:
        rank = np.sum(info <= info_at_pos) / len(info)
    else:
        rank = 0.5

    # Feature 6: How well the WT scores relative to all 20 AAs at this position
    wt_rank = np.sum(row <= pssm_wt) / 20.0

    return [pssm_wt, pssm_mut, delta_pssm, info_at_pos, rank, wt_rank]


def main():
    print("=" * 70)
    print("GENERATING PSSM CONSERVATION FEATURES")
    print("=" * 70)

    os.makedirs(PSSM_DIR, exist_ok=True)

    # Check BLAST database exists
    if not os.path.exists(BLASTDB + ".pdb"):
        print(f"ERROR: SwissProt BLAST database not found at {BLASTDB}")
        print("Run: cd blastdb && update_blastdb.pl --decompress swissprot")
        sys.exit(1)

    # Collect unique proteins
    proteins = collect_unique_proteins()

    # Check how many PSSMs we already have cached
    already_cached = sum(1 for pid in proteins if os.path.exists(os.path.join(PSSM_DIR, f"{pid}.pssm")))
    to_compute = len(proteins) - already_cached
    print(f"\nPSSMs cached: {already_cached}, to compute: {to_compute}")

    # Run PSI-BLAST for each protein
    conservation_cache = {}
    failed = 0
    success = 0
    start_time = time.time()

    for i, (protein_id, sequence) in enumerate(proteins.items()):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 1)
            remaining = (len(proteins) - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{len(proteins)}] Processing {protein_id} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        result = run_psiblast(protein_id, sequence)

        if result is not None:
            sh = seq_hash(sequence)
            conservation_cache[(protein_id, sh)] = result
            # Also store by protein_id alone for easy lookup
            conservation_cache[protein_id] = result
            success += 1
        else:
            failed += 1

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Success: {success}, Failed: {failed}")

    # Save cache
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(conservation_cache, f)
    print(f"Saved conservation cache to {CACHE_PATH}")
    print(f"Cache size: {os.path.getsize(CACHE_PATH) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
