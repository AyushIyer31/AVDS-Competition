"""Fetch PETase and related enzyme structures from RCSB PDB."""

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DATA_URL = "https://data.rcsb.org/rest/v1/core/entry"
RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry"

KNOWN_PETASE_IDS = [
    "5XJH",  # IsPETase from Ideonella sakaiensis
    "6EQE",  # High-res IsPETase (0.92A)
    "5XG0",  # IsPETase variant
    "6ANE",  # PETase structure
    "5YNS",  # PETase R280A mutant
    "7CGA",  # p38gamma (control)
    "6IJ6",  # ThermoPETase S121E/D186H/R280A
    "4EB0",  # Leaf-branch compost cutinase
    "6EQD",  # IsPETase long wavelength
    "6EQF",  # IsPETase P212121
    "6EQG",  # IsPETase P21
    "6EQH",  # IsPETase C2221
    "6IJ3",  # PETase S121D/D186H
    "6IJ4",  # PETase S121E/D186H
    "6IJ5",  # PETase P181A
]

# Enzyme family classification
ENZYME_FAMILIES = {
    "PETase": ["5XJH", "6EQE", "5XG0", "6ANE", "5YNS", "6IJ6", "6EQD", "6EQF", "6EQG", "6EQH", "6IJ3", "6IJ4", "6IJ5", "6QGC", "8J17", "8J5N"],
    "Cutinase": ["4EB0", "29CU", "4CG1", "4CG2", "4CG3", "7QJO", "7QJP", "7QJR", "8BRA", "8BRB", "8Z2G", "8Z2H", "8Z2I", "8Z2J", "8Z2K"],
    "LCC Variant": ["7VVC", "7VVE", "7W1N", "7W44", "7W45", "8CMV", "8JMO", "8JMP", "8QRJ"],
    "Ancestral PETase": ["8ETX", "8ETY"],
    "Thermostable Hydrolase": ["7YKO", "7YKP", "7YKQ", "8GZD"],
    "Novel PET Hydrolase": ["7PZJ", "7Z6B", "8OTU", "9LMT", "9LMU", "9LMV", "9LMW"],
}

def _classify_enzyme(pdb_id: str) -> str:
    """Classify enzyme by PDB ID into a family."""
    for family, ids in ENZYME_FAMILIES.items():
        if pdb_id in ids:
            return family
    return "Related Hydrolase"

# In-memory cache
_cache: list[dict] = []
_cache_time: float = 0
_CACHE_TTL = 600  # 10 minutes


def search_petase_structures(max_results: int = 150) -> list[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "or",
            "nodes": [
                {"type": "terminal", "service": "full_text", "parameters": {"value": "PETase plastic degrading"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "polyethylene terephthalate hydrolase"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "cutinase PET degradation"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "cutinase thermostable"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "polyester hydrolase"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "esterase plastic biodegradation"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "Thermobifida fusca cutinase"}},
                {"type": "terminal", "service": "full_text", "parameters": {"value": "leaf branch compost cutinase LCC"}},
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
        return list(KNOWN_PETASE_IDS)


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
            "family": _classify_enzyme(pdb_id),
        }
    except Exception:
        return {"pdb_id": pdb_id, "title": "Unknown", "organism": "Unknown", "resolution": None, "family": _classify_enzyme(pdb_id)}


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
    with ThreadPoolExecutor(max_workers=16) as executor:
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
