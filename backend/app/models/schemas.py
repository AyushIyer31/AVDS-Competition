from pydantic import BaseModel


class SequenceInput(BaseModel):
    sequence: str
    name: str = "query_sequence"


class OptimizationRequest(BaseModel):
    sequence: str
    num_candidates: int = 10
    optimization_steps: int = 50
    target_temperature: float = 60.0


class MutationCandidate(BaseModel):
    rank: int
    sequence: str
    mutations: list[str]
    predicted_stability_score: float
    predicted_activity_score: float
    combined_score: float


class OptimizationResponse(BaseModel):
    original_sequence: str
    candidates: list[MutationCandidate]
    latent_space_summary: dict


class EmbeddingResponse(BaseModel):
    sequence: str
    embedding_dim: int
    mean_embedding: list[float]


class PDBSearchResult(BaseModel):
    pdb_id: str
    title: str
    organism: str
    resolution: float | None
    sequence: str
