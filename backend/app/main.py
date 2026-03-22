"""FastAPI backend for PETase ML optimization."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models.schemas import (
    SequenceInput,
    OptimizationRequest,
    OptimizationResponse,
    EmbeddingResponse,
    MutationCandidate,
    PDBSearchResult,
)
from .services import pdb_fetcher, esm_engine, latent_optimizer
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
        mutations = esm_engine.scan_beneficial_mutations(req.sequence, top_k=30)
        return {"sequence_length": len(req.sequence), "beneficial_mutations": mutations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize", response_model=OptimizationResponse)
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
