class PDBResult {
  final String pdbId;
  final String title;
  final String organism;
  final double? resolution;
  final String sequence;

  PDBResult({
    required this.pdbId,
    required this.title,
    required this.organism,
    this.resolution,
    required this.sequence,
  });

  factory PDBResult.fromJson(Map<String, dynamic> json) {
    return PDBResult(
      pdbId: json['pdb_id'] as String,
      title: json['title'] as String,
      organism: json['organism'] as String? ?? 'Unknown',
      resolution: (json['resolution'] as num?)?.toDouble(),
      sequence: json['sequence'] as String,
    );
  }
}

class MutationCandidate {
  final int rank;
  final String sequence;
  final List<String> mutations;
  final double predictedStabilityScore;
  final double predictedActivityScore;
  final double combinedScore;

  MutationCandidate({
    required this.rank,
    required this.sequence,
    required this.mutations,
    required this.predictedStabilityScore,
    required this.predictedActivityScore,
    required this.combinedScore,
  });

  factory MutationCandidate.fromJson(Map<String, dynamic> json) {
    return MutationCandidate(
      rank: json['rank'] as int,
      sequence: json['sequence'] as String,
      mutations: List<String>.from(json['mutations'] as List),
      predictedStabilityScore:
          (json['predicted_stability_score'] as num).toDouble(),
      predictedActivityScore:
          (json['predicted_activity_score'] as num).toDouble(),
      combinedScore: (json['combined_score'] as num).toDouble(),
    );
  }
}

class OptimizationResult {
  final String originalSequence;
  final List<MutationCandidate> candidates;
  final Map<String, dynamic> latentSpaceSummary;

  OptimizationResult({
    required this.originalSequence,
    required this.candidates,
    required this.latentSpaceSummary,
  });

  factory OptimizationResult.fromJson(Map<String, dynamic> json) {
    return OptimizationResult(
      originalSequence: json['original_sequence'] as String,
      candidates: (json['candidates'] as List)
          .map((c) => MutationCandidate.fromJson(c as Map<String, dynamic>))
          .toList(),
      latentSpaceSummary: json['latent_space_summary'] as Map<String, dynamic>,
    );
  }
}

class BeneficialMutation {
  final int position;
  final String wildType;
  final String mutant;
  final double score;
  final String label;

  BeneficialMutation({
    required this.position,
    required this.wildType,
    required this.mutant,
    required this.score,
    required this.label,
  });

  factory BeneficialMutation.fromJson(Map<String, dynamic> json) {
    return BeneficialMutation(
      position: json['position'] as int,
      wildType: json['wild_type'] as String,
      mutant: json['mutant'] as String,
      score: (json['score'] as num).toDouble(),
      label: json['label'] as String,
    );
  }
}
