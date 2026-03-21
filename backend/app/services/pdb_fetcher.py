"""Fetch PETase and related enzyme structures from RCSB PDB."""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry"

KNOWN_PETASE_IDS = [
    "5XJH",  # IsPETase from Ideonella sakaiensis
    "6EQE",  # Thermostable PETase variant
    "5XG0",  # IsPETase W159H/S238F
    "6ANE",  # PETase double mutant
    "5YNS",  # PETase S121E/D186H/R280A
    "7CGA",  # FAST-PETase
    "6IJ6",  # Leaf-branch compost cutinase (LCC)
    "4EB0",  # Cutinase (related hydrolase)
]

# In-memory cache
_cache: list[dict] = []
_cache_time: float = 0
_CACHE_TTL = 600  # 10 minutes


def search_petase_structures(max_results: int = 50) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "or",
            "nodes": [
                {"type": "terminal", "service": "full_text", "parameters": {"value": "PETase plastic degrading"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "polyethylene terephthalate hydrolase"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "cutinase PET degradation"}},
            ],
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"], "paginate": {"start": 0, "rows": max_results}},
    }
    try:
        resp = requests.post(RCSB_SEARCH_URL, json=query, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [hit["identifier"] for hit in data.get("result_set", [])]
    except Exception:
        return KNOWN_PETASE_IDS


def fetch_entry_metadata(pdb_id: str) -> dict:
    try:
        resp = requests.get(f"{RCSB_DATA_URL}/{pdb_id}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "pdb_id": pdb_id,
            "title": data.get("struct", {}).get("title", "Unknown"),
            "organism": (
                data.get("rcsb_entry_info", {}).get("deposited_organism", ["Unknown"])[0]
                if data.get("rcsb_entry_info", {}).get("deposited_organism")
                else "Unknown"
            ),
            "resolution": data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0],
        }
    except Exception:
        return {"pdb_id": pdb_id, "title": "Unknown", "organism": "Unknown", "resolution": None}


def fetch_sequence(pdb_id: str) -> str:
    try:
        resp = requests.get(f"{RCSB_FASTA_URL}/{pdb_id}", timeout=10)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        return "".join(line.strip() for line in lines if not line.startswith(">"))
    except Exception:
        return ""


def _fetch_single_entry(pdb_id: str) -> dict | None:
    """Fetch metadata + sequence for one PDB entry."""
    meta = fetch_entry_metadata(pdb_id)
    seq = fetch_sequence(pdb_id)
    if seq:
        meta["sequence"] = seq
        return meta
    return None


def fetch_all_petase_data() -> list[dict]:
    """Fetch all PETase data with parallel requests and caching."""
    global _cache, _cache_time

    if _cache and (time.time() - _cache_time) < _CACHE_TTL:
        return _cache

    pdb_ids = search_petase_structures()
    all_ids = list(dict.fromkeys(KNOWN_PETASE_IDS + pdb_ids))

    results = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(_fetch_single_entry, pid): pid for pid in all_ids}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Sort: known IDs first, then alphabetically
    known_set = set(KNOWN_PETASE_IDS)
    results.sort(key=lambda r: (0 if r["pdb_id"] in known_set else 1, r["pdb_id"]))

    _cache = results
    _cache_time = time.time()
    return results
