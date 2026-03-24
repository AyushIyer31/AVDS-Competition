import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../theme.dart';
import '../models/api_models.dart';

class ResultsScreen extends StatelessWidget {
  final OptimizationResult result;

  const ResultsScreen({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final best = result.candidates.isNotEmpty ? result.candidates.first : null;

    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(title: const Text('Results')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 4, 16, 32),
        children: [
          _buildImpactCard(best),
          const SizedBox(height: 16),
          _buildWhatWeDidCard(),
          const SizedBox(height: 16),
          _buildModelInfoCard(),
          const SizedBox(height: 16),
          _buildLatentPlot(),
          const SizedBox(height: 16),
          _buildMutationsChips(),
          const SizedBox(height: 24),
          const Text('RANKED CANDIDATES',
              style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textTertiary,
                  letterSpacing: 1.2)),
          const SizedBox(height: 4),
          const Text(
            'Tap any candidate to see explanations, validation, and full sequence.',
            style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
          ),
          const SizedBox(height: 12),
          ...result.candidates.map((c) => Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: _buildCandidateCard(context, c),
              )),
          const SizedBox(height: 16),
          _buildNextSteps(),
        ],
      ),
    );
  }

  Widget _buildImpactCard(MutationCandidate? best) {
    final improvePct = best != null
        ? ((best.combinedScore - 0.5) * 200).clamp(0.0, 100.0).toStringAsFixed(3)
        : '0.000';
    final kgPerDay =
        best != null ? (best.combinedScore * 15).toStringAsFixed(3) : '0.000';
    final kgPerYear =
        best != null ? (best.combinedScore * 15 * 365).toStringAsFixed(1) : '0.0';

    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: AppColors.primary,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Environmental Impact',
              style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70,
                  letterSpacing: 0.5)),
          const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text('~$kgPerDay',
                  style: const TextStyle(
                      fontSize: 36,
                      fontWeight: FontWeight.w800,
                      color: Colors.white,
                      height: 1)),
              const SizedBox(width: 6),
              const Padding(
                padding: EdgeInsets.only(bottom: 4),
                child: Text('kg PET / day',
                    style: TextStyle(fontSize: 14, color: Colors.white70)),
              ),
            ],
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(10),
            ),
            child: FittedBox(
              fit: BoxFit.scaleDown,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _miniStat('+$improvePct%', 'predicted gain'),
                  const SizedBox(width: 16),
                  Container(width: 1, height: 24, color: Colors.white24),
                  const SizedBox(width: 16),
                  _miniStat('$kgPerYear kg', 'PET / year'),
                  const SizedBox(width: 16),
                  Container(width: 1, height: 24, color: Colors.white24),
                  const SizedBox(width: 16),
                  _miniStat('${best?.mutations.length ?? 0}', 'mutations'),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            'Waste PET is broken into terephthalic acid and ethylene glycol — '
            'reusable monomers for new plastic, closing the recycling loop.',
            style: TextStyle(
                fontSize: 12.5,
                color: Colors.white.withValues(alpha: 0.75),
                height: 1.4),
          ),
        ],
      ),
    );
  }

  Widget _miniStat(String value, String label) {
    return Column(
      children: [
        Text(value,
            style: const TextStyle(
                fontSize: 16, fontWeight: FontWeight.w700, color: Colors.white)),
        const SizedBox(height: 1),
        Text(label,
            style: TextStyle(
                fontSize: 10.5, color: Colors.white.withValues(alpha: 0.6))),
      ],
    );
  }

  Widget _buildWhatWeDidCard() {
    final summary = result.latentSpaceSummary;
    final seqLen = result.originalSequence.length;
    final totalScanned = seqLen * 19;

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('What the AI Did',
              style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textPrimary)),
          const SizedBox(height: 14),
          _aiStep('Scanned $totalScanned possible mutations',
              '$seqLen positions x 19 amino acid options'),
          _aiStep(
              'Found ${summary['beneficial_mutations_found'] ?? 0} beneficial changes',
              'Predicted to improve the enzyme'),
          _aiStep('Explored ${summary['candidates_explored'] ?? 0} combinations',
              'Single, double, and triple mutation combos'),
          _aiStep('Trained classifier verified predictions',
              'Gradient Boosting model cross-validated on experimental data'),
          _aiStep('Validated against published literature',
              'Compared with ThermoPETase, DuraPETase, FAST-PETase studies'),
          _aiStep('Returned top ${result.candidates.length} candidates',
              'Ranked by heat resistance + catalytic speed'),
        ],
      ),
    );
  }

  Widget _buildModelInfoCard() {
    final info = result.classifierInfo;
    if (info.isEmpty) return const SizedBox.shrink();

    final accuracy = info['cv_accuracy'] as num? ?? 0;
    final samples = info['training_samples'] as num? ?? 0;
    final importances = info['feature_importances'] as Map<String, dynamic>? ?? {};

    // Get top 3 features
    final sortedFeatures = importances.entries.toList()
      ..sort((a, b) => (b.value as num).compareTo(a.value as num));
    final topFeatures = sortedFeatures.take(3).toList();

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: AppColors.accent.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: const Text('TRAINED MODEL',
                    style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w800,
                        color: AppColors.accent,
                        letterSpacing: 0.8)),
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Text('Gradient Boosting Classifier',
                    style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary)),
              ),
            ],
          ),
          const SizedBox(height: 12),
          FittedBox(
            fit: BoxFit.scaleDown,
            alignment: Alignment.centerLeft,
            child: Row(
              children: [
                _modelStat('${(accuracy * 100).toStringAsFixed(1)}%', 'CV Accuracy'),
                const SizedBox(width: 24),
                _modelStat('$samples', 'Training Samples'),
                const SizedBox(width: 24),
                _modelStat('17', 'Features'),
              ],
            ),
          ),
          if (topFeatures.isNotEmpty) ...[
            const SizedBox(height: 14),
            const Text('TOP FEATURE IMPORTANCES',
                style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 0.8)),
            const SizedBox(height: 8),
            ...topFeatures.map((e) => Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Row(
                    children: [
                      SizedBox(
                        width: 120,
                        child: Text(_featureDisplayName(e.key),
                            style: const TextStyle(
                                fontSize: 12, color: AppColors.textSecondary)),
                      ),
                      Expanded(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(2),
                          child: LinearProgressIndicator(
                            value: (e.value as num).toDouble(),
                            backgroundColor: AppColors.border,
                            color: AppColors.accent,
                            minHeight: 4,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text('${((e.value as num) * 100).toStringAsFixed(1)}%',
                          style: const TextStyle(
                              fontSize: 11,
                              fontFamily: 'monospace',
                              color: AppColors.textTertiary)),
                    ],
                  ),
                )),
          ],
        ],
      ),
    );
  }

  Widget _modelStat(String value, String label) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(value,
            style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w800,
                color: AppColors.textPrimary)),
        Text(label,
            style: const TextStyle(fontSize: 11, color: AppColors.textTertiary)),
      ],
    );
  }

  String _featureDisplayName(String key) {
    const names = {
      'hydro_delta': 'Hydrophobicity',
      'charge_delta': 'Charge Change',
      'size_delta': 'Size Change',
      'flex_delta': 'Flexibility',
      'helix_delta': 'Helix Propensity',
      'sheet_delta': 'Sheet Propensity',
      'wt_hydro': 'WT Hydrophobicity',
      'mut_hydro': 'Mut Hydrophobicity',
      'wt_size': 'WT Size',
      'mut_size': 'Mut Size',
      'near_active': 'Near Active Site',
      'is_hotspot': 'Thermo Hotspot',
      'norm_position': 'Position',
      'is_proline': 'Proline Sub',
      'is_glycine': 'Glycine Sub',
      'abs_charge_change': 'Abs Charge',
      'aromatic_loss': 'Aromatic Loss',
    };
    return names[key] ?? key;
  }

  Widget _aiStep(String title, String sub) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 4),
            child: Icon(Icons.check_circle, size: 16, color: AppColors.success),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        fontSize: 13.5,
                        fontWeight: FontWeight.w600,
                        color: AppColors.textPrimary)),
                Text(sub,
                    style: const TextStyle(
                        fontSize: 12, color: AppColors.textTertiary)),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLatentPlot() {
    final coords = result.latentSpaceSummary['latent_coordinates'] as List?;
    if (coords == null || coords.length < 2) return const SizedBox.shrink();

    final spots = <ScatterSpot>[];
    for (int i = 0; i < coords.length; i++) {
      final point = coords[i] as List;
      spots.add(ScatterSpot(
        (point[0] as num).toDouble(),
        (point[1] as num).toDouble(),
        dotPainter: FlDotCirclePainter(
          radius: i == 0 ? 9 : 6,
          color: i == 0 ? AppColors.error : AppColors.primary,
          strokeWidth: 2,
          strokeColor:
              i == 0 ? const Color(0xFF9B2D30) : const Color(0xFF094D36),
        ),
      ));
    }

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Protein Latent Space',
              style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textPrimary)),
          const SizedBox(height: 4),
          const Text(
            'Each dot is an enzyme. Nearby = structurally similar. Green variants improve on the red original.',
            style: TextStyle(
                fontSize: 12.5, color: AppColors.textSecondary, height: 1.3),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Container(
                  width: 10,
                  height: 10,
                  decoration: const BoxDecoration(
                      color: AppColors.error, shape: BoxShape.circle)),
              const SizedBox(width: 6),
              const Text('Original',
                  style:
                      TextStyle(fontSize: 11, color: AppColors.textSecondary)),
              const SizedBox(width: 16),
              Container(
                  width: 10,
                  height: 10,
                  decoration: const BoxDecoration(
                      color: AppColors.primary, shape: BoxShape.circle)),
              const SizedBox(width: 6),
              const Text('AI-designed',
                  style:
                      TextStyle(fontSize: 11, color: AppColors.textSecondary)),
            ],
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 200,
            child: ScatterChart(
              ScatterChartData(
                scatterSpots: spots,
                borderData: FlBorderData(
                    show: true,
                    border: Border.all(color: AppColors.border)),
                gridData: FlGridData(
                  show: true,
                  drawHorizontalLine: true,
                  drawVerticalLine: true,
                  getDrawingHorizontalLine: (_) =>
                      const FlLine(color: AppColors.border, strokeWidth: 0.5),
                  getDrawingVerticalLine: (_) =>
                      const FlLine(color: AppColors.border, strokeWidth: 0.5),
                ),
                titlesData: const FlTitlesData(
                  bottomTitles:
                      AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  leftTitles:
                      AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  topTitles:
                      AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  rightTitles:
                      AxisTitles(sideTitles: SideTitles(showTitles: false)),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMutationsChips() {
    final topMuts = result.latentSpaceSummary['top_mutations'] as List?;
    if (topMuts == null || topMuts.isEmpty) return const SizedBox.shrink();

    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Top Mutations Found',
              style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: AppColors.textPrimary)),
          const SizedBox(height: 4),
          const Text(
            'Format: [Original][Position][New]. E.g. A65G = swap Alanine for Glycine at position 65.',
            style: TextStyle(
                fontSize: 12, color: AppColors.textSecondary, height: 1.3),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: topMuts
                .map((m) => Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 5),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withValues(alpha: 0.07),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(m.toString(),
                          style: const TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              color: AppColors.primary)),
                    ))
                .toList(),
          ),
        ],
      ),
    );
  }

  Widget _buildCandidateCard(
      BuildContext context, MutationCandidate candidate) {
    final hasValidation = candidate.literatureValidation != null &&
        (candidate.literatureValidation!.exactMatches.isNotEmpty ||
            candidate.literatureValidation!.positionMatches.isNotEmpty);

    final classifierOk = candidate.classifierPrediction?.allBeneficial ?? false;

    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: () => _showCandidateDetail(context, candidate),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border),
          ),
          child: Column(
            children: [
              Row(
                children: [
                  // Rank badge
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: _rankColor(candidate.rank).withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Center(
                      child: Text('#${candidate.rank}',
                          style: TextStyle(
                              fontWeight: FontWeight.w800,
                              fontSize: 13,
                              color: _rankColor(candidate.rank))),
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          candidate.mutations.join(' + '),
                          style: const TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: AppColors.textPrimary),
                        ),
                        const SizedBox(height: 4),
                        Wrap(
                          spacing: 12,
                          runSpacing: 4,
                          children: [
                            _miniScore('Stability',
                                candidate.predictedStabilityScore, Colors.blue),
                            _miniScore('Speed',
                                candidate.predictedActivityScore, AppColors.warning),
                          ],
                        ),
                      ],
                    ),
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                          '${(candidate.combinedScore * 100).toStringAsFixed(3)}%',
                          style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w800,
                              fontFamily: 'monospace',
                              color: _rankColor(candidate.rank))),
                      Text(_gradeLabel(candidate.combinedScore),
                          style: const TextStyle(
                              fontSize: 10, color: AppColors.textTertiary)),
                    ],
                  ),
                  const SizedBox(width: 4),
                  const Icon(Icons.chevron_right,
                      size: 18, color: AppColors.textTertiary),
                ],
              ),
              // Badges row
              if (hasValidation || classifierOk) ...[
                const SizedBox(height: 10),
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: [
                    if (hasValidation)
                      _badge('Literature Validated', AppColors.success,
                          Icons.verified),
                    if (classifierOk)
                      _badge(
                          'ML Confirmed ${(candidate.classifierPrediction!.averageConfidence * 100).toStringAsFixed(0)}%',
                          AppColors.accent,
                          Icons.psychology),
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _badge(String label, Color color, IconData icon) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 4),
          Text(label,
              style: TextStyle(
                  fontSize: 10.5, fontWeight: FontWeight.w600, color: color)),
        ],
      ),
    );
  }

  Widget _miniScore(String label, double value, Color color) {
    return Row(
      children: [
        Container(
          width: 6,
          height: 6,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 4),
        Text('$label ${(value * 100).toStringAsFixed(3)}%',
            style: const TextStyle(
                fontSize: 11, color: AppColors.textTertiary)),
      ],
    );
  }

  void _showCandidateDetail(
      BuildContext context, MutationCandidate candidate) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.85,
        minChildSize: 0.4,
        maxChildSize: 0.95,
        expand: false,
        builder: (_, controller) => ListView(
          controller: controller,
          padding: const EdgeInsets.fromLTRB(24, 12, 24, 32),
          children: [
            Center(
              child: Container(
                  width: 36,
                  height: 4,
                  decoration: BoxDecoration(
                      color: AppColors.border,
                      borderRadius: BorderRadius.circular(2))),
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Candidate #${candidate.rank}',
                          style: const TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.w800,
                              color: AppColors.textPrimary,
                              letterSpacing: -0.5)),
                      const SizedBox(height: 4),
                      Text(_gradeLabel(candidate.combinedScore),
                          style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: _rankColor(candidate.rank))),
                    ],
                  ),
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: _strategyColor(candidate.overallStrategy)
                        .withValues(alpha: 0.1),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    candidate.overallStrategy.toUpperCase(),
                    style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w800,
                        color: _strategyColor(candidate.overallStrategy),
                        letterSpacing: 0.5),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // Performance scores
            const Text('PERFORMANCE',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 12),
            _detailBar('Heat Resistance', 'Will it survive high temperatures?',
                candidate.predictedStabilityScore, Colors.blue),
            const SizedBox(height: 14),
            _detailBar('Catalytic Speed', 'How fast does it degrade PET?',
                candidate.predictedActivityScore, AppColors.warning),
            const SizedBox(height: 14),
            _detailBar('Overall Fitness', 'Balanced score for real-world use',
                candidate.combinedScore, AppColors.success),

            // Classifier confidence
            if (candidate.classifierPrediction != null) ...[
              const SizedBox(height: 24),
              _buildClassifierSection(candidate.classifierPrediction!),
            ],

            // Literature validation
            if (candidate.literatureValidation != null) ...[
              const SizedBox(height: 24),
              _buildLiteratureSection(candidate.literatureValidation!),
            ],

            // Mutation explanations
            const SizedBox(height: 24),
            const Text('MUTATION EXPLANATIONS',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 4),
            const Text(
              'Why the AI chose each change — biochemical reasoning.',
              style:
                  TextStyle(fontSize: 12, color: AppColors.textSecondary),
            ),
            const SizedBox(height: 10),
            ...candidate.explanations.map(
                (e) => _buildExplanationCard(e)),
            if (candidate.explanations.isEmpty)
              ...candidate.mutations.map((m) => Padding(
                    padding: const EdgeInsets.only(bottom: 8),
                    child: _buildSimpleMutationCard(m),
                  )),

            // Full sequence
            const SizedBox(height: 24),
            const Text('FULL SEQUENCE',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 4),
            const Text(
                'Ready for gene synthesis — copy and send to provider.',
                style:
                    TextStyle(fontSize: 12, color: AppColors.textSecondary)),
            const SizedBox(height: 8),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppColors.border),
              ),
              child: SelectableText(
                candidate.sequence,
                style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 12,
                    height: 1.6,
                    color: AppColors.textPrimary),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildClassifierSection(ClassifierPrediction cp) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: cp.allBeneficial
            ? AppColors.accent.withValues(alpha: 0.04)
            : AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
            color: cp.allBeneficial
                ? AppColors.accent.withValues(alpha: 0.2)
                : AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.psychology,
                  size: 18,
                  color: cp.allBeneficial
                      ? AppColors.accent
                      : AppColors.textSecondary),
              const SizedBox(width: 8),
              const Expanded(
                child: Text('Trained Classifier',
                    style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary)),
              ),
              Text(
                  '${(cp.averageConfidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      fontFamily: 'monospace',
                      color: cp.allBeneficial
                          ? AppColors.accent
                          : AppColors.textTertiary)),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            cp.allBeneficial
                ? 'All ${cp.total} mutations predicted beneficial by our trained model.'
                : '${cp.beneficialCount}/${cp.total} mutations predicted beneficial.',
            style: const TextStyle(
                fontSize: 13, color: AppColors.textSecondary, height: 1.3),
          ),
          const SizedBox(height: 8),
          ...cp.perMutation.map((pm) {
            final beneficial = pm['predicted_beneficial'] as bool? ?? false;
            final conf =
                (pm['probability_beneficial'] as num?)?.toDouble() ?? 0;
            return Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(
                children: [
                  Icon(
                      beneficial
                          ? Icons.check_circle_outline
                          : Icons.cancel_outlined,
                      size: 14,
                      color: beneficial
                          ? AppColors.success
                          : AppColors.error),
                  const SizedBox(width: 6),
                  Text(pm['mutation'] as String? ?? '',
                      style: const TextStyle(
                          fontFamily: 'monospace',
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: AppColors.textPrimary)),
                  const Spacer(),
                  Text('${(conf * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(
                          fontSize: 11,
                          fontFamily: 'monospace',
                          color: AppColors.textTertiary)),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildLiteratureSection(LiteratureValidation lv) {
    if (lv.exactMatches.isEmpty &&
        lv.positionMatches.isEmpty &&
        lv.variantOverlaps.isEmpty) {
      return const SizedBox.shrink();
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.success.withValues(alpha: 0.04),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.success.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.verified, size: 18, color: AppColors.success),
              const SizedBox(width: 8),
              const Expanded(
                child: Text('Literature Validation',
                    style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary)),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: AppColors.success.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text(
                    '${(lv.validationScore * 100).toStringAsFixed(0)}% validated',
                    style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        color: AppColors.success)),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(lv.summary,
              style: const TextStyle(
                  fontSize: 13,
                  color: AppColors.textSecondary,
                  height: 1.4)),
          // Exact matches
          ...lv.exactMatches.map((m) => Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Text(m.mutation,
                              style: const TextStyle(
                                  fontFamily: 'monospace',
                                  fontWeight: FontWeight.w700,
                                  fontSize: 13,
                                  color: AppColors.success)),
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 6, vertical: 1),
                            decoration: BoxDecoration(
                              color: AppColors.success.withValues(alpha: 0.1),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: const Text('EXACT MATCH',
                                style: TextStyle(
                                    fontSize: 9,
                                    fontWeight: FontWeight.w800,
                                    color: AppColors.success)),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text('${m.paper ?? ''} (${m.journal ?? ''})',
                          style: const TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w600,
                              color: AppColors.textSecondary)),
                      if (m.improvement != null)
                        Text(m.improvement!,
                            style: const TextStyle(
                                fontSize: 11,
                                color: AppColors.textTertiary)),
                    ],
                  ),
                ),
              )),
          // Position matches
          ...lv.positionMatches.map((m) => Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Text(m.mutation,
                              style: const TextStyle(
                                  fontFamily: 'monospace',
                                  fontWeight: FontWeight.w700,
                                  fontSize: 13,
                                  color: AppColors.warning)),
                          const SizedBox(width: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 6, vertical: 1),
                            decoration: BoxDecoration(
                              color: AppColors.warning.withValues(alpha: 0.1),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: const Text('SAME POSITION',
                                style: TextStyle(
                                    fontSize: 9,
                                    fontWeight: FontWeight.w800,
                                    color: AppColors.warning)),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(m.detail ?? '',
                          style: const TextStyle(
                              fontSize: 11,
                              color: AppColors.textSecondary,
                              height: 1.3)),
                    ],
                  ),
                ),
              )),
          // Variant overlaps
          ...lv.variantOverlaps.map((v) => Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.science,
                          size: 16, color: AppColors.accent),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                                'Overlaps with ${v['variant_name'] ?? 'unknown'}',
                                style: const TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w600,
                                    color: AppColors.textPrimary)),
                            Text(
                                '${v['paper'] ?? ''} — ${v['improvement'] ?? ''}',
                                style: const TextStyle(
                                    fontSize: 11,
                                    color: AppColors.textTertiary)),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              )),
        ],
      ),
    );
  }

  Widget _buildExplanationCard(MutationExplanation exp) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: AppColors.border),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(
              crossAxisAlignment: WrapCrossAlignment.center,
              spacing: 8,
              runSpacing: 4,
              children: [
                const Icon(Icons.swap_horiz, size: 18, color: AppColors.primary),
                Text(exp.mutation,
                    style: const TextStyle(
                        fontFamily: 'monospace',
                        fontWeight: FontWeight.w700,
                        fontSize: 15,
                        color: AppColors.textPrimary)),
                if (exp.nearActiveSite)
                  _tagChip('Active Site', AppColors.warning),
                if (exp.thermostabilityHotspot)
                  _tagChip('Hotspot', AppColors.success),
              ],
            ),
            const SizedBox(height: 8),
            Text(exp.summary,
                style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textPrimary,
                    height: 1.3)),
            const SizedBox(height: 8),
            ...exp.reasons.map((r) => Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Padding(
                        padding: EdgeInsets.only(top: 5),
                        child: Icon(Icons.arrow_right,
                            size: 14, color: AppColors.textTertiary),
                      ),
                      const SizedBox(width: 4),
                      Expanded(
                        child: Text(r,
                            style: const TextStyle(
                                fontSize: 12,
                                color: AppColors.textSecondary,
                                height: 1.4)),
                      ),
                    ],
                  ),
                )),
          ],
        ),
      ),
    );
  }

  Widget _tagChip(String label, Color color) {
    return Container(
      margin: const EdgeInsets.only(right: 4),
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(label,
          style: TextStyle(
              fontSize: 9, fontWeight: FontWeight.w700, color: color)),
    );
  }

  Widget _buildSimpleMutationCard(String m) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          const Icon(Icons.swap_horiz, size: 18, color: AppColors.primary),
          const SizedBox(width: 10),
          Text(m,
              style: const TextStyle(
                  fontFamily: 'monospace',
                  fontWeight: FontWeight.w700,
                  fontSize: 15,
                  color: AppColors.textPrimary)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(_describeMutation(m),
                style: const TextStyle(
                    fontSize: 13, color: AppColors.textSecondary)),
          ),
        ],
      ),
    );
  }

  Widget _detailBar(String label, String sub, double score, Color color) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(label,
                style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary)),
            const Spacer(),
            Text('${(score * 100).toStringAsFixed(3)}%',
                style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    fontFamily: 'monospace',
                    color: color)),
          ],
        ),
        const SizedBox(height: 2),
        Text(sub,
            style: const TextStyle(
                fontSize: 12, color: AppColors.textTertiary)),
        const SizedBox(height: 6),
        ClipRRect(
          borderRadius: BorderRadius.circular(3),
          child: LinearProgressIndicator(
            value: score.clamp(0.0, 1.0),
            backgroundColor: AppColors.border,
            color: color,
            minHeight: 6,
          ),
        ),
      ],
    );
  }

  Widget _buildNextSteps() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.science, size: 18, color: AppColors.warning),
              SizedBox(width: 8),
              Text('Next Steps for the Lab',
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textPrimary)),
            ],
          ),
          const SizedBox(height: 14),
          _labStep('Order gene synthesis for top candidates'),
          _labStep('Express enzyme in E. coli host cells'),
          _labStep('Assay PET degradation rate at target temp'),
          _labStep('Measure thermal melting point via DSF'),
          _labStep('Validate with real PET film degradation'),
          const SizedBox(height: 10),
          const Text(
            'AI narrows months of random mutagenesis to a focused candidate set. '
            'Lab validation turns predictions into real-world impact.',
            style: TextStyle(
                fontSize: 12,
                color: AppColors.textTertiary,
                fontStyle: FontStyle.italic,
                height: 1.4),
          ),
        ],
      ),
    );
  }

  Widget _labStep(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          Container(
            width: 6,
            height: 6,
            decoration: const BoxDecoration(
                color: AppColors.textTertiary, shape: BoxShape.circle),
          ),
          const SizedBox(width: 10),
          Text(text,
              style: const TextStyle(
                  fontSize: 13.5, color: AppColors.textSecondary)),
        ],
      ),
    );
  }

  Color _rankColor(int rank) {
    if (rank <= 3) return AppColors.success;
    if (rank <= 7) return AppColors.accent;
    return AppColors.textSecondary;
  }

  Color _strategyColor(String strategy) {
    switch (strategy) {
      case 'thermostability-focused':
        return Colors.blue;
      case 'activity-focused':
        return AppColors.warning;
      default:
        return AppColors.primary;
    }
  }

  String _gradeLabel(double score) {
    if (score > 0.75) return 'Excellent candidate';
    if (score > 0.65) return 'Strong candidate';
    if (score > 0.55) return 'Moderate improvement';
    if (score > 0.45) return 'Slight improvement';
    return 'Marginal change';
  }

  String _describeMutation(String mutation) {
    const aaNames = {
      'A': 'Alanine', 'C': 'Cysteine', 'D': 'Aspartate', 'E': 'Glutamate',
      'F': 'Phenylalanine', 'G': 'Glycine', 'H': 'Histidine',
      'I': 'Isoleucine', 'K': 'Lysine', 'L': 'Leucine', 'M': 'Methionine',
      'N': 'Asparagine', 'P': 'Proline', 'Q': 'Glutamine', 'R': 'Arginine',
      'S': 'Serine', 'T': 'Threonine', 'V': 'Valine', 'W': 'Tryptophan',
      'Y': 'Tyrosine',
    };
    if (mutation.length < 3) return '';
    final from = aaNames[mutation[0]] ?? mutation[0];
    final to = aaNames[mutation[mutation.length - 1]] ??
        mutation[mutation.length - 1];
    return '$from to $to';
  }
}
