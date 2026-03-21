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
            'Tap any candidate to see details and full sequence.',
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
    final improvePct =
        best != null ? ((best.combinedScore - 0.5) * 200).clamp(0.0, 100.0).toStringAsFixed(3) : '0.000';
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
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _miniStat('+$improvePct%', 'predicted gain'),
                Container(width: 1, height: 24, color: Colors.white24),
                _miniStat('$kgPerYear kg', 'PET / year'),
                Container(width: 1, height: 24, color: Colors.white24),
                _miniStat('${best?.mutations.length ?? 0}', 'mutations'),
              ],
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
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: Colors.white)),
        const SizedBox(height: 1),
        Text(label,
            style: TextStyle(
                fontSize: 10.5,
                color: Colors.white.withValues(alpha: 0.6))),
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
          _aiStep('Returned top ${result.candidates.length} candidates',
              'Ranked by heat resistance + catalytic speed'),
        ],
      ),
    );
  }

  Widget _aiStep(String title, String sub) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(top: 4),
            child:
                Icon(Icons.check_circle, size: 16, color: AppColors.success),
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
          strokeColor: i == 0 ? const Color(0xFF9B2D30) : const Color(0xFF094D36),
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
                fontSize: 12.5,
                color: AppColors.textSecondary,
                height: 1.3),
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
                  style: TextStyle(fontSize: 11, color: AppColors.textSecondary)),
              const SizedBox(width: 16),
              Container(
                  width: 10,
                  height: 10,
                  decoration: const BoxDecoration(
                      color: AppColors.primary, shape: BoxShape.circle)),
              const SizedBox(width: 6),
              const Text('AI-designed',
                  style: TextStyle(fontSize: 11, color: AppColors.textSecondary)),
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
                  bottomTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  leftTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  topTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  rightTitles: AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
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
            style: TextStyle(fontSize: 12, color: AppColors.textSecondary, height: 1.3),
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

  Widget _buildCandidateCard(BuildContext context, MutationCandidate candidate) {
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
          child: Row(
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
                    Row(
                      children: [
                        _miniScore('Stability', candidate.predictedStabilityScore,
                            Colors.blue),
                        const SizedBox(width: 12),
                        _miniScore('Speed', candidate.predictedActivityScore,
                            AppColors.warning),
                      ],
                    ),
                  ],
                ),
              ),
              Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text('${(candidate.combinedScore * 100).toStringAsFixed(3)}%',
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
        ),
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

  void _showCandidateDetail(BuildContext context, MutationCandidate candidate) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        minChildSize: 0.4,
        maxChildSize: 0.92,
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
            const SizedBox(height: 24),

            // Score bars
            const Text('PERFORMANCE',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 12),
            _detailBar('Heat Resistance',
                'Will it survive high temperatures?',
                candidate.predictedStabilityScore, Colors.blue),
            const SizedBox(height: 14),
            _detailBar('Catalytic Speed',
                'How fast does it degrade PET?',
                candidate.predictedActivityScore, AppColors.warning),
            const SizedBox(height: 14),
            _detailBar('Overall Fitness',
                'Balanced score for real-world use',
                candidate.combinedScore, AppColors.success),

            const SizedBox(height: 24),

            // Mutations
            const Text('MUTATIONS',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 10),
            ...candidate.mutations.map((m) => Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.swap_horiz,
                            size: 18, color: AppColors.primary),
                        const SizedBox(width: 10),
                        Text(m,
                            style: const TextStyle(
                                fontFamily: 'monospace',
                                fontWeight: FontWeight.w700,
                                fontSize: 15,
                                color: AppColors.textPrimary)),
                        const SizedBox(width: 10),
                        Text(_describeMutation(m),
                            style: const TextStyle(
                                fontSize: 13,
                                color: AppColors.textSecondary)),
                      ],
                    ),
                  ),
                )),

            const SizedBox(height: 24),

            // Sequence
            const Text('FULL SEQUENCE',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 4),
            const Text('Ready for gene synthesis — copy and send to provider.',
                style: TextStyle(fontSize: 12, color: AppColors.textSecondary)),
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

  Widget _detailBar(
      String label, String sub, double score, Color color) {
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
                    fontSize: 15, fontWeight: FontWeight.w700, fontFamily: 'monospace', color: color)),
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
    final to = aaNames[mutation[mutation.length - 1]] ?? mutation[mutation.length - 1];
    return '$from to $to';
  }
}
