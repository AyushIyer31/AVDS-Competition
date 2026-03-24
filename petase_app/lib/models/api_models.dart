class PDBResult {
  final String pdbId;
  final String title;
  final String organism;
  final double? resolution;
  final String sequence;
  final String family;

  PDBResult({
    required this.pdbId,
    required this.title,
    required this.organism,
    this.resolution,
    required this.sequence,
    this.family = 'Related Hydrolase',
  });

  factory PDBResult.fromJson(Map<String, dynamic> json) {
    return PDBResult(
      pdbId: json['pdb_id'] as String,
      title: json['title'] as String,
      organism: json['organism'] as String? ?? 'Unknown',
      resolution: (json['resolution'] as num?)?.toDouble(),
      sequence: json['sequence'] as String,
      family: json['family'] as String? ?? 'Related Hydrolase',
    );
  }
}

class MutationExplanation {
  final String mutation;
  final String fromAA;
  final String toAA;
  final String fromCategory;
  final String toCategory;
  final int position;
  final String summary;
  final List<String> reasons;
  final List<String> effects;
  final bool nearActiveSite;
  final bool thermostabilityHotspot;
  final String riskLevel;
  final double esmScore;

  MutationExplanation({
    required this.mutation,
    required this.fromAA,
    required this.toAA,
    required this.fromCategory,
    required this.toCategory,
    required this.position,
    required this.summary,
    required this.reasons,
    required this.effects,
    required this.nearActiveSite,
    required this.thermostabilityHotspot,
    required this.riskLevel,
    required this.esmScore,
  });

  factory MutationExplanation.fromJson(Map<String, dynamic> json) {
    return MutationExplanation(
      mutation: json['mutation'] as String? ?? '',
      fromAA: json['from_aa'] as String? ?? '',
      toAA: json['to_aa'] as String? ?? '',
      fromCategory: json['from_category'] as String? ?? '',
      toCategory: json['to_category'] as String? ?? '',
      position: json['position'] as int? ?? 0,
      summary: json['summary'] as String? ?? '',
      reasons: List<String>.from(json['reasons'] as List? ?? []),
      effects: List<String>.from(json['effects'] as List? ?? []),
      nearActiveSite: json['near_active_site'] as bool? ?? false,
      thermostabilityHotspot: json['thermostability_hotspot'] as bool? ?? false,
      riskLevel: json['risk_level'] as String? ?? 'low',
      esmScore: (json['esm_score'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class LiteratureMatch {
  final String mutation;
  final String matchType;
  final String? paper;
  final String? journal;
  final String? improvement;
  final String? detail;
  final String? variantName;
  final String? literatureMutation;

  LiteratureMatch({
    required this.mutation,
    required this.matchType,
    this.paper,
    this.journal,
    this.improvement,
    this.detail,
    this.variantName,
    this.literatureMutation,
  });

  factory LiteratureMatch.fromJson(Map<String, dynamic> json) {
    return LiteratureMatch(
      mutation: json['mutation'] as String? ?? '',
      matchType: json['match_type'] as String? ?? 'novel',
      paper: json['paper'] as String?,
      journal: json['journal'] as String?,
      improvement: json['improvement'] as String?,
      detail: json['detail'] as String?,
      variantName: json['variant_name'] as String?,
      literatureMutation: json['literature_mutation'] as String?,
    );
  }
}

class LiteratureValidation {
  final List<LiteratureMatch> exactMatches;
  final List<LiteratureMatch> positionMatches;
  final List<LiteratureMatch> novelPredictions;
  final List<Map<String, dynamic>> variantOverlaps;
  final double validationScore;
  final String summary;

  LiteratureValidation({
    required this.exactMatches,
    required this.positionMatches,
    required this.novelPredictions,
    required this.variantOverlaps,
    required this.validationScore,
    required this.summary,
  });

  factory LiteratureValidation.fromJson(Map<String, dynamic> json) {
    return LiteratureValidation(
      exactMatches: (json['exact_matches'] as List? ?? [])
          .map((m) => LiteratureMatch.fromJson(m as Map<String, dynamic>))
          .toList(),
      positionMatches: (json['position_matches'] as List? ?? [])
          .map((m) => LiteratureMatch.fromJson(m as Map<String, dynamic>))
          .toList(),
      novelPredictions: (json['novel_predictions'] as List? ?? [])
          .map((m) => LiteratureMatch.fromJson(m as Map<String, dynamic>))
          .toList(),
      variantOverlaps: (json['variant_overlaps'] as List? ?? [])
          .map((v) => Map<String, dynamic>.from(v as Map))
          .toList(),
      validationScore: (json['validation_score'] as num?)?.toDouble() ?? 0.0,
      summary: json['summary'] as String? ?? '',
    );
  }
}

class ClassifierPrediction {
  final bool allBeneficial;
  final int beneficialCount;
  final int total;
  final double averageConfidence;
  final List<Map<String, dynamic>> perMutation;

  ClassifierPrediction({
    required this.allBeneficial,
    required this.beneficialCount,
    required this.total,
    required this.averageConfidence,
    required this.perMutation,
  });

  factory ClassifierPrediction.fromJson(Map<String, dynamic> json) {
    return ClassifierPrediction(
      allBeneficial: json['all_beneficial'] as bool? ?? false,
      beneficialCount: json['beneficial_count'] as int? ?? 0,
      total: json['total'] as int? ?? 0,
      averageConfidence:
          (json['average_confidence'] as num?)?.toDouble() ?? 0.0,
      perMutation: (json['per_mutation'] as List? ?? [])
          .map((m) => Map<String, dynamic>.from(m as Map))
          .toList(),
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
  final List<MutationExplanation> explanations;
  final String overallStrategy;
  final LiteratureValidation? literatureValidation;
  final ClassifierPrediction? classifierPrediction;

  MutationCandidate({
    required this.rank,
    required this.sequence,
    required this.mutations,
    required this.predictedStabilityScore,
    required this.predictedActivityScore,
    required this.combinedScore,
    this.explanations = const [],
    this.overallStrategy = 'balanced',
    this.literatureValidation,
    this.classifierPrediction,
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
      explanations: (json['explanations'] as List? ?? [])
          .map((e) =>
              MutationExplanation.fromJson(e as Map<String, dynamic>))
          .toList(),
      overallStrategy: json['overall_strategy'] as String? ?? 'balanced',
      literatureValidation: json['literature_validation'] != null
          ? LiteratureValidation.fromJson(
              json['literature_validation'] as Map<String, dynamic>)
          : null,
      classifierPrediction: json['classifier_prediction'] != null
          ? ClassifierPrediction.fromJson(
              json['classifier_prediction'] as Map<String, dynamic>)
          : null,
    );
  }
}

class OptimizationResult {
  final String originalSequence;
  final List<MutationCandidate> candidates;
  final Map<String, dynamic> latentSpaceSummary;
  final Map<String, dynamic> classifierInfo;

  OptimizationResult({
    required this.originalSequence,
    required this.candidates,
    required this.latentSpaceSummary,
    this.classifierInfo = const {},
  });

  factory OptimizationResult.fromJson(Map<String, dynamic> json) {
    return OptimizationResult(
      originalSequence: json['original_sequence'] as String,
      candidates: (json['candidates'] as List)
          .map((c) => MutationCandidate.fromJson(c as Map<String, dynamic>))
          .toList(),
      latentSpaceSummary:
          json['latent_space_summary'] as Map<String, dynamic>,
      classifierInfo:
          (json['classifier_info'] as Map<String, dynamic>?) ?? {},
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
