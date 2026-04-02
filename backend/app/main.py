"""FastAPI backend for PETase ML optimization."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from .models.schemas import (
    SequenceInput,
    OptimizationRequest,
    OptimizationResponse,
    EmbeddingResponse,
    MutationCandidate,
    PDBSearchResult,
)
from .services import pdb_fetcher, latent_optimizer
from .services import explainability, literature_validation, trained_classifier

app = FastAPI(
    title="PETase ML Optimizer",
    description="ML-driven enzyme engineering for plastic-degrading PETase enzymes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default IsPETase wild-type sequence (Ideonella sakaiensis, PDB: 5XJH)
ISPETASE_SEQUENCE = (
    "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAG"
    "TVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALR"
    "QVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTL"
    "IFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDT"
    "RYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
)


@app.get("/")
async def root():
    return {
        "service": "PETase ML Optimizer",
        "version": "1.0.0",
        "endpoints": [
            "/pdb/search",
            "/pdb/sequence/{pdb_id}",
            "/esm/embedding",
            "/optimize",
            "/health",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/pdb/search", response_model=list[PDBSearchResult])
async def search_pdb():
    """Search RCSB PDB for PETase-related structures."""
    try:
        results = pdb_fetcher.fetch_all_petase_data()
        return [
            PDBSearchResult(
                pdb_id=r["pdb_id"],
                title=r["title"],
                organism=r.get("organism", "Unknown"),
                resolution=r.get("resolution"),
                sequence=r["sequence"],
                family=r.get("family", "Related Hydrolase"),
            )
            for r in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pdb/sequence/{pdb_id}")
async def get_pdb_sequence(pdb_id: str):
    """Fetch sequence for a specific PDB ID."""
    sequence = pdb_fetcher.fetch_sequence(pdb_id.upper())
    if not sequence:
        raise HTTPException(status_code=404, detail=f"No sequence found for {pdb_id}")
    meta = pdb_fetcher.fetch_entry_metadata(pdb_id.upper())
    return {"pdb_id": pdb_id.upper(), "sequence": sequence, **meta}


@app.post("/esm/embedding", response_model=EmbeddingResponse)
async def compute_embedding(req: SequenceInput):
    """Compute ESM-2 embedding for a protein sequence."""
    if not req.sequence or len(req.sequence) < 10:
        raise HTTPException(status_code=400, detail="Sequence must be at least 10 residues")
    if len(req.sequence) > 1000:
        raise HTTPException(status_code=400, detail="Sequence must be under 1000 residues")

    try:
        from .services import esm_engine
        embedding = esm_engine.get_sequence_embedding(req.sequence)
        return EmbeddingResponse(
            sequence=req.sequence,
            embedding_dim=len(embedding),
            mean_embedding=embedding.tolist(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/esm/mutations")
async def scan_mutations(req: SequenceInput):
    """Scan for beneficial single-point mutations using ESM-2."""
    if not req.sequence or len(req.sequence) < 10:
        raise HTTPException(status_code=400, detail="Sequence must be at least 10 residues")

    try:
        from .services import esm_engine
        mutations = esm_engine.scan_beneficial_mutations(req.sequence, top_k=30)
        return {"sequence_length": len(req.sequence), "beneficial_mutations": mutations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_petase(req: OptimizationRequest):
    """Run full latent space optimization to generate improved PETase candidates."""
    sequence = req.sequence or ISPETASE_SEQUENCE
    if len(sequence) < 10:
        raise HTTPException(status_code=400, detail="Sequence must be at least 10 residues")

    try:
        result = latent_optimizer.optimize(
            sequence=sequence,
            num_candidates=req.num_candidates,
            optimization_steps=req.optimization_steps,
            target_temp=req.target_temperature,
        )
        return OptimizationResponse(
            original_sequence=result["original_sequence"],
            candidates=[MutationCandidate(**c) for c in result["candidates"]],
            latent_space_summary=result["latent_space_summary"],
            classifier_info=result.get("classifier_info", {}),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/mutation")
async def explain_mutation(req: SequenceInput):
    """Explain a single mutation. Pass mutation as the 'name' field (e.g. S121E)."""
    mut_str = req.name
    if len(mut_str) < 3:
        raise HTTPException(status_code=400, detail="Mutation format: S121E")
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    position = int(mut_str[1:-1]) - 1
    result = explainability.explain_mutation(wt_aa, mut_aa, position)
    return result


@app.post("/explain/candidate")
async def explain_candidate_mutations(req: SequenceInput):
    """Explain all mutations in a candidate. Pass comma-separated mutations as 'name'."""
    mutations = [m.strip() for m in req.name.split(",") if m.strip()]
    result = explainability.explain_candidate(mutations)
    return result


@app.get("/literature/known-mutations")
async def known_mutations():
    """Return all experimentally validated PETase mutations from literature."""
    return {
        "mutations": literature_validation.get_all_known_mutations(),
        "named_variants": literature_validation.NAMED_VARIANTS,
    }


@app.post("/literature/validate")
async def validate_against_literature(req: SequenceInput):
    """Validate predicted mutations against published experiments. Pass comma-separated mutations as 'name'."""
    mutations = [m.strip() for m in req.name.split(",") if m.strip()]
    return literature_validation.validate_mutations(mutations)


@app.get("/classifier/info")
async def classifier_info():
    """Return trained classifier model info and metrics."""
    metrics = trained_classifier.get_training_metrics()
    return metrics


@app.post("/classifier/predict")
async def classifier_predict(req: SequenceInput):
    """Predict mutation effect using trained classifier. Pass comma-separated mutations as 'name'."""
    mutations = [m.strip() for m in req.name.split(",") if m.strip()]
    return trained_classifier.predict_candidate_mutations(mutations)


@app.get("/default-sequence")
async def default_sequence():
    """Return the default IsPETase wild-type sequence."""
    return {"name": "IsPETase (Ideonella sakaiensis)", "pdb_id": "5XJH", "sequence": ISPETASE_SEQUENCE}


# ---------- 3D Structure Viewer ----------
# Cache predicted structures to avoid re-calling ESMFold
_STRUCTURE_CACHE: dict[str, str] = {}

# Known PDB IDs for preset sequences (avoid ESMFold when unnecessary)
_LCC_SEQUENCE = (
    "SNPYQRGPNPTRSALTADGPFSVATYTVSRLSVSGFGGGVIYYPTGTSLTFGGIAMSPGYTADASSL"
    "AWLGRRLASHGFVVLVINTNSRFDYPDSRASQLSAALNYLRTSSPSAVRARLDANRLAVAGHSMGGG"
    "GTLRIAEQNPSLKAAVPLTPWHTDKTFNTSVPVLIVGAEADTVAPVSQHAIPFYQNLPSTTPKVYV"
    "ELDNASHFAPNSNNAAISVYTISWMKLWVDNDTRYRQFLCNVNDPALSDFRTNNRHCQ"
)

_KNOWN_SEQUENCE_PDBS: dict[str, str] = {
    ISPETASE_SEQUENCE: "5XJH",
    _LCC_SEQUENCE: "4EB0",
}


from pydantic import BaseModel as _BaseModel

class StructureRequest(_BaseModel):
    sequence: str
    mutations: str = ""
    title: str = ""
    original_sequence: str = ""


@app.post("/api/structure-viewer", response_class=HTMLResponse)
async def structure_viewer(req: StructureRequest):
    """Return an interactive 3Dmol.js HTML page.

    Uses known PDB structures when available, otherwise predicts
    the structure using ESMFold (Meta's protein structure prediction model).
    """
    sequence = req.sequence.strip().upper()
    original = req.original_sequence.strip().upper() if req.original_sequence else sequence

    # Check cache first
    pdb_data = _STRUCTURE_CACHE.get(original) or _STRUCTURE_CACHE.get(sequence)
    source = "cached"

    if not pdb_data:
        pdb_id = None

        # 1. Try known PDB for the original wild-type sequence
        pdb_id = _KNOWN_SEQUENCE_PDBS.get(original)

        # 2. Try subsequence match against known sequences
        if not pdb_id:
            for known_seq, kid in _KNOWN_SEQUENCE_PDBS.items():
                if original in known_seq or known_seq in original:
                    pdb_id = kid
                    break

        # 3. Try similarity match — if sequences differ by < 5%, use the same PDB
        if not pdb_id:
            for known_seq, kid in _KNOWN_SEQUENCE_PDBS.items():
                if len(original) == len(known_seq):
                    diffs = sum(1 for a, b in zip(original, known_seq) if a != b)
                    if diffs / len(original) < 0.05:
                        pdb_id = kid
                        break
                # Also check the candidate sequence itself
                if len(sequence) == len(known_seq):
                    diffs = sum(1 for a, b in zip(sequence, known_seq) if a != b)
                    if diffs / len(sequence) < 0.05:
                        pdb_id = kid
                        break

        # 4. Try fetching from RCSB search by sequence
        if not pdb_id:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    search_query = {
                        "query": {
                            "type": "terminal",
                            "service": "sequence",
                            "parameters": {
                                "evalue_cutoff": 0.1,
                                "identity_cutoff": 0.9,
                                "sequence_type": "protein",
                                "value": original[:400],
                            }
                        },
                        "return_type": "entry",
                        "request_options": {"results_content_type": ["experimental"], "return_all_hits": False}
                    }
                    resp = await client.post(
                        "https://search.rcsb.org/rcsbsearch/v2/query",
                        json=search_query,
                    )
                    if resp.status_code == 200:
                        hits = resp.json().get("result_set", [])
                        if hits:
                            pdb_id = hits[0]["identifier"]
            except Exception:
                pass

        if pdb_id:
            # Fetch crystal structure from RCSB
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
                if resp.status_code == 200:
                    pdb_data = resp.text
                    source = f"PDB: {pdb_id} (crystal structure)"

        if not pdb_data:
            raise HTTPException(
                status_code=404,
                detail="No structure found. Try using a known enzyme from the presets."
            )

        # Cache for future requests
        _STRUCTURE_CACHE[original] = pdb_data

    # Amino acid name lookup
    _AA_NAMES = {
        'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic Acid',
        'C': 'Cysteine', 'E': 'Glutamic Acid', 'Q': 'Glutamine', 'G': 'Glycine',
        'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
        'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
        'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine',
    }

    # Parse mutations like "S121E,D186H,R280A"
    mut_list = [m.strip() for m in req.mutations.split(",") if m.strip()]
    mut_positions = []
    mut_labels = []
    mut_details = []  # (label, wt_name, mut_name, position)
    for m in mut_list:
        try:
            pos = int(m[1:-1])
            wt_aa = m[0]
            mut_aa = m[-1]
            mut_positions.append(pos)
            mut_labels.append(m)
            mut_details.append({
                "label": m,
                "position": pos,
                "from_code": wt_aa,
                "to_code": mut_aa,
                "from_name": _AA_NAMES.get(wt_aa, wt_aa),
                "to_name": _AA_NAMES.get(mut_aa, mut_aa),
            })
        except (ValueError, IndexError):
            pass

    # Catalytic residues for IsPETase-like enzymes
    catalytic = [160, 206, 237]

    # Build JS for highlighting — make mutations very prominent
    mut_selections_js = ""
    for i, pos in enumerate(mut_positions):
        wt_aa = mut_labels[i][0]
        mut_aa = mut_labels[i][-1]
        mut_selections_js += f"""
        // Mutation {mut_labels[i]}: big sphere + thick stick + pulsing glow
        viewer.addStyle({{resi: {pos}}}, {{
            stick: {{color: '#FF6B35', radius: 0.25}},
            sphere: {{color: '#FF6B35', opacity: 0.55, radius: 1.2}}
        }});
        // Bright label with mutation detail
        viewer.addLabel("{mut_labels[i]}  ({wt_aa}\u2192{mut_aa})", {{
            position: {{resi: {pos}}},
            backgroundColor: '#FF6B35',
            fontColor: 'white',
            fontSize: 14,
            fontWeight: 'bold',
            padding: 4,
            borderRadius: 6,
            borderColor: '#FF8855',
            borderThickness: 1.5,
            showBackground: true
        }});
        """

    catalytic_js = ""
    for pos in catalytic:
        catalytic_js += f"""
        viewer.addStyle({{resi: {pos}}}, {{
            stick: {{color: '#0FB5A2', radius: 0.2}},
            sphere: {{color: '#0FB5A2', opacity: 0.35, radius: 0.9}}
        }});
        viewer.addLabel("Catalytic {pos}", {{
            position: {{resi: {pos}}},
            backgroundColor: '#0FB5A2',
            fontColor: 'white',
            fontSize: 11,
            padding: 3,
            borderRadius: 5,
            showBackground: true
        }});
        """

    display_title = req.title if req.title else "3D Structure"

    # Build mutation details HTML
    mut_detail_html = ""
    if mut_details:
        rows = ""
        for md in mut_details:
            rows += f"""<div class="mut-row">
              <span class="mut-badge">{md['label']}</span>
              <span class="mut-desc">
                <span class="aa-from">{md['from_name']}</span>
                <span class="mut-arrow">&rarr;</span>
                <span class="aa-to">{md['to_name']}</span>
                <span class="mut-pos">Position {md['position']}</span>
              </span>
            </div>"""
        mut_detail_html = f"""<div id="mutations-panel">
          <div class="panel-title">Mutations vs. Wild-Type ({len(mut_details)} change{'s' if len(mut_details) != 1 else ''})</div>
          {rows}
        </div>"""
    else:
        mut_detail_html = """<div id="mutations-panel">
          <div class="panel-title">No mutations — Wild-type structure</div>
        </div>"""

    # Escape PDB data for JS
    pdb_escaped = pdb_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #1A1A2E;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    color: white;
    overflow-x: hidden;
    overflow-y: auto;
  }}
  #header {{
    padding: 12px 16px 8px;
    background: rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }}
  #header h2 {{
    font-size: 16px;
    font-weight: 600;
    color: #E0E8F0;
    margin-bottom: 8px;
  }}

  #source-tag {{
    font-size: 10px;
    color: #6B8AB5;
    font-style: italic;
    margin-top: 4px;
  }}

  /* --- Color Key (below viewer) --- */
  #color-key {{
    padding: 14px 16px;
    background: rgba(255,255,255,0.05);
    border-top: 1px solid rgba(255,255,255,0.08);
  }}
  .key-title {{
    font-size: 13px;
    font-weight: 600;
    color: #C0D0E0;
    margin-bottom: 10px;
    letter-spacing: 0.3px;
  }}
  .key-grid {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .key-item {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
  }}
  .key-swatch {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    flex-shrink: 0;
  }}
  .key-swatch-sphere {{
    width: 16px;
    height: 16px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .key-spectrum {{
    width: 48px;
    height: 16px;
    border-radius: 4px;
    background: linear-gradient(90deg, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000);
    flex-shrink: 0;
  }}
  .key-label {{
    font-size: 13px;
    color: #AAB8C8;
    font-weight: 500;
  }}
  .key-desc {{
    font-size: 11px;
    color: #667788;
  }}

  /* --- Viewer --- */
  #viewer-container {{
    width: 100vw;
    height: 55vh;
    min-height: 320px;
  }}

  /* --- Mutations Panel --- */
  #mutations-panel {{
    padding: 12px 16px 16px;
    background: rgba(255,255,255,0.03);
    border-top: 1px solid rgba(255,255,255,0.08);
  }}
  .panel-title {{
    font-size: 13px;
    font-weight: 600;
    color: #C0D0E0;
    margin-bottom: 10px;
    letter-spacing: 0.3px;
  }}
  .mut-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    margin-bottom: 6px;
    background: rgba(255,107,53,0.08);
    border: 1px solid rgba(255,107,53,0.2);
    border-radius: 8px;
  }}
  .mut-badge {{
    background: #FF6B35;
    color: white;
    font-size: 13px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 5px;
    font-family: 'SF Mono', Menlo, monospace;
    letter-spacing: 0.5px;
    white-space: nowrap;
  }}
  .mut-desc {{
    display: flex;
    align-items: center;
    gap: 5px;
    flex-wrap: wrap;
    font-size: 12px;
  }}
  .aa-from {{
    color: #FF8888;
    font-weight: 500;
    text-decoration: line-through;
    text-decoration-color: rgba(255,136,136,0.4);
  }}
  .mut-arrow {{
    color: #556677;
    font-size: 14px;
  }}
  .aa-to {{
    color: #88DD88;
    font-weight: 600;
  }}
  .mut-pos {{
    color: #667788;
    font-size: 11px;
    margin-left: 4px;
  }}

  #tip {{
    text-align: center;
    padding: 8px;
    font-size: 11px;
    color: rgba(255,255,255,0.3);
  }}
</style>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
<div id="header">
  <h2>{display_title}</h2>
  <div id="source-tag">{source}</div>
</div>
<div id="viewer-container"></div>
{mut_detail_html}
<div id="color-key">
  <div class="key-title">Color Key</div>
  <div class="key-grid">
    <div class="key-item">
      <span class="key-spectrum"></span>
      <span><span class="key-label">Protein Backbone</span><br><span class="key-desc">Rainbow spectrum from N-terminus (blue) to C-terminus (red)</span></span>
    </div>
    <div class="key-item">
      <span class="key-swatch-sphere" style="background:#FF6B35"></span>
      <span><span class="key-label">AI-Predicted Mutations</span><br><span class="key-desc">Orange spheres &amp; sticks — positions modified by the optimizer</span></span>
    </div>
    <div class="key-item">
      <span class="key-swatch-sphere" style="background:#0FB5A2"></span>
      <span><span class="key-label">Catalytic Triad</span><br><span class="key-desc">Aqua spheres — active site residues (Ser160, His206, Asp237)</span></span>
    </div>
  </div>
</div>
<div id="tip">Pinch to zoom &middot; Drag to rotate &middot; Two-finger drag to pan</div>
<script>
let viewer = $3Dmol.createViewer("viewer-container", {{
  backgroundColor: "0x1A1A2E",
  antialias: true,
  cartoonQuality: 10
}});

let pdbData = `{pdb_escaped}`;
viewer.addModel(pdbData, "pdb");

// Base style: rainbow spectrum cartoon (blue=N-terminus → red=C-terminus)
viewer.setStyle({{}}, {{
  cartoon: {{
    color: "spectrum",
    opacity: 0.85,
    thickness: 0.25
  }}
}});

// Highlight mutations (orange sticks)
{mut_selections_js}

// Highlight catalytic residues (aqua sticks)
{catalytic_js}

viewer.zoomTo();
viewer.spin(false);
viewer.render();

// Auto-spin slowly on load, stop on interaction
let spinning = true;
viewer.spin("y", 0.5);
document.getElementById("viewer-container").addEventListener("touchstart", function() {{
  if (spinning) {{ viewer.spin(false); spinning = false; }}
}});
document.getElementById("viewer-container").addEventListener("mousedown", function() {{
  if (spinning) {{ viewer.spin(false); spinning = false; }}
}});
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
