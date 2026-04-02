"""Step 1: Compute ESM-2 embeddings for all FireProtDB sequences."""
import pandas as pd, numpy as np, pickle, os, torch, esm

MODEL_DIR = os.path.join(os.path.dirname(__file__), "backend", "app", "trained_models")
ESM_CACHE_PATH = os.path.join(MODEL_DIR, "esm2_embeddings.pkl")

# Load existing cache
esm_cache = {}
if os.path.exists(ESM_CACHE_PATH):
    with open(ESM_CACHE_PATH, 'rb') as f:
        esm_cache = pickle.load(f)
    print(f"Loaded existing cache: {len(esm_cache)} sequences")

# Get unique sequences from FireProtDB
df = pd.read_csv('fireprotdb_data/fireprot_upload/csvs/4_fireprotDB_bestpH.csv')
seq_map = {}
for _, r in df.iterrows():
    uid = str(r.get('uniprot_id', '')).strip()
    seq = str(r.get('sequence', '')).strip()
    if uid and uid != 'nan' and seq and len(seq) > 10:
        seq_map[uid] = seq

to_compute = {k: v for k, v in seq_map.items() if k not in esm_cache}
print(f"Total sequences: {len(seq_map)}, to compute: {len(to_compute)}")

if not to_compute:
    print("All done!")
    exit(0)

# Load ESM-2 model
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
print("Model loaded!")

for i, (uid, seq) in enumerate(to_compute.items()):
    seq_trunc = seq[:800]
    try:
        data = [("protein", seq_trunc)]
        _, _, tokens = batch_converter(data)
        with torch.no_grad():
            results = model(tokens, repr_layers=[12])
        reps = results["representations"][12][0, 1:len(seq_trunc)+1].cpu().numpy()
        embeddings = {}
        for pos_idx in range(len(seq_trunc)):
            embeddings[pos_idx + 1] = reps[pos_idx]
        esm_cache[uid] = embeddings
    except Exception as e:
        print(f"  Failed {uid} (len={len(seq)}): {e}")
        esm_cache[uid] = {}

    if (i+1) % 10 == 0 or (i+1) == len(to_compute):
        print(f"  {i+1}/{len(to_compute)} done")
    if (i+1) % 25 == 0:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(ESM_CACHE_PATH, 'wb') as f:
            pickle.dump(esm_cache, f)

# Final save
os.makedirs(MODEL_DIR, exist_ok=True)
with open(ESM_CACHE_PATH, 'wb') as f:
    pickle.dump(esm_cache, f)
print(f"\nDone! Cached {len(esm_cache)} sequences to {ESM_CACHE_PATH}")
