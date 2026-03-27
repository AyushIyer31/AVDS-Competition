import 'package:flutter/material.dart';
import '../theme.dart';
import '../services/api_service.dart';
import 'pdb_browser_screen.dart';
import 'optimize_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _backendOnline = false;
  bool _checking = true;

  @override
  void initState() {
    super.initState();
    _checkBackend();
  }

  Future<void> _checkBackend() async {
    setState(() => _checking = true);
    final online = await ApiService.checkHealth();
    setState(() {
      _backendOnline = online;
      _checking = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Top bar
              Row(
                children: [
                  Container(
                    width: 36,
                    height: 36,
                    decoration: BoxDecoration(
                      color: AppColors.primary,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Center(
                      child: Text('P',
                          style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w800,
                              fontSize: 18)),
                    ),
                  ),
                  const SizedBox(width: 10),
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('PET Lab',
                          style: TextStyle(
                              fontSize: 17,
                              fontWeight: FontWeight.w700,
                              color: AppColors.textPrimary,
                              letterSpacing: -0.3)),
                      Text('Enzyme Engineering Platform',
                          style: TextStyle(
                              fontSize: 12,
                              color: AppColors.textTertiary,
                              letterSpacing: -0.1)),
                    ],
                  ),
                  const Spacer(),
                  _buildStatusIndicator(),
                ],
              ),

              const SizedBox(height: 24),

              // Hero section
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [Color(0xFF1A7FC4), Color(0xFF0E9B8E)],
                  ),
                  borderRadius: BorderRadius.circular(18),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Text('AI-Powered Enzyme Engineering',
                          style: TextStyle(
                              fontSize: 11,
                              fontWeight: FontWeight.w700,
                              color: Colors.white70,
                              letterSpacing: 0.5)),
                    ),
                    const SizedBox(height: 14),
                    const Text(
                      'Designing enzymes\nthat eat plastic.',
                      style: TextStyle(
                        fontSize: 26,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                        height: 1.15,
                        letterSpacing: -0.5,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      'Using ESM-2 and a trained Gradient Boosting classifier '
                      'to engineer thermostable PETase enzymes for '
                      'industrial-scale plastic recycling.',
                      style: TextStyle(
                        fontSize: 13.5,
                        color: Colors.white.withValues(alpha: 0.8),
                        height: 1.45,
                      ),
                    ),
                    const SizedBox(height: 20),
                    // Stats row
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 14),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const FittedBox(
                        fit: BoxFit.scaleDown,
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.spaceAround,
                          children: [
                            _StatColumn(value: '400M+', unit: 'tons/yr', label: 'plastic produced'),
                            SizedBox(width: 16),
                            _StatDivider(),
                            SizedBox(width: 16),
                            _StatColumn(value: '<10%', unit: '', label: 'gets recycled'),
                            SizedBox(width: 16),
                            _StatDivider(),
                            SizedBox(width: 16),
                            _StatColumn(value: '1000+', unit: 'years', label: 'to decompose'),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Section label
              const Text('TOOLS',
                  style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textTertiary,
                      letterSpacing: 1.2)),
              const SizedBox(height: 10),

              // Action cards
              _ActionCard(
                icon: Icons.biotech,
                title: 'Browse Enzyme Database',
                description: 'Explore plastic-degrading enzymes from the Protein Data Bank',
                trailing: '107 structures',
                onTap: () => Navigator.push(context,
                    MaterialPageRoute(builder: (_) => const PDBBrowserScreen())),
              ),
              const SizedBox(height: 10),
              _ActionCard(
                icon: Icons.auto_fix_high,
                title: 'Design Better Enzyme',
                description: 'AI finds mutations for heat resistance and faster catalysis',
                trailing: 'ESM-2',
                onTap: () => Navigator.push(context,
                    MaterialPageRoute(builder: (_) => const OptimizeScreen())),
              ),
              const SizedBox(height: 10),
              _ActionCard(
                icon: Icons.school,
                title: 'How It Works',
                description: 'The 5-step AI pipeline from sequence to optimized enzyme',
                trailing: '',
                onTap: () => _showPipeline(context),
              ),

              const SizedBox(height: 28),

              // Problem statement
              const Text('THE PROBLEM',
                  style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textTertiary,
                      letterSpacing: 1.2)),
              const SizedBox(height: 10),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Natural PETase enzymes can break down PET plastic, but they denature above 40 degrees C. '
                      'Industrial recycling requires 60-70 degrees C.',
                      style: TextStyle(
                          fontSize: 14,
                          color: AppColors.textPrimary,
                          height: 1.5),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Our AI scans 6,000+ possible mutations in seconds to find the ones that make the enzyme survive '
                      'industrial heat while maintaining catalytic speed. This replaces months of lab trial-and-error.',
                      style: TextStyle(
                          fontSize: 13,
                          color: AppColors.textSecondary,
                          height: 1.5),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 28),
              Center(
                child: Text(
                  'AVDS Hackathon 2026',
                  style: TextStyle(
                      color: AppColors.textTertiary,
                      fontSize: 11,
                      letterSpacing: 0.5),
                ),
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatusIndicator() {
    if (_checking) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.grey[100],
          borderRadius: BorderRadius.circular(20),
        ),
        child: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
                width: 10,
                height: 10,
                child: CircularProgressIndicator(strokeWidth: 1.5)),
            SizedBox(width: 6),
            Text('Connecting...',
                style: TextStyle(fontSize: 11, color: AppColors.textTertiary)),
          ],
        ),
      );
    }

    return GestureDetector(
      onTap: _checkBackend,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: _backendOnline
              ? AppColors.success.withValues(alpha: 0.08)
              : AppColors.error.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(20),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 7,
              height: 7,
              decoration: BoxDecoration(
                color: _backendOnline ? AppColors.success : AppColors.error,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 6),
            Text(
              _backendOnline ? 'ML Ready' : 'Offline',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: _backendOnline ? AppColors.success : AppColors.error,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showPipeline(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.75,
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
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 24),
            const Text('How the Pipeline Works',
                style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -0.5,
                    color: AppColors.textPrimary)),
            const SizedBox(height: 6),
            const Text(
              'From raw protein data to engineered enzymes in 5 steps',
              style: TextStyle(fontSize: 14, color: AppColors.textSecondary),
            ),
            const SizedBox(height: 28),
            _pipelineStep(1, 'Collect Known Enzymes',
                'Pull real plastic-degrading enzyme structures from the RCSB Protein Data Bank — 200,000+ structures worldwide.'),
            _pipelineStep(2, 'Understand with AI',
                'Facebook\'s ESM-2 (650M parameters, trained on 250M proteins) reads the amino acid sequence and learns how each residue contributes to function.'),
            _pipelineStep(3, 'Find Helpful Mutations',
                'The AI scores every possible amino acid swap — 6,000+ options for a typical enzyme. Positive score = predicted improvement.'),
            _pipelineStep(4, 'Combine and Rank',
                'Top mutations are combined into multi-mutation candidates scored for heat resistance, catalytic speed, and overall fitness.'),
            _pipelineStep(5, 'Output for the Lab',
                'Top-ranked candidates are returned with full sequences ready for gene synthesis and experimental validation.'),
          ],
        ),
      ),
    );
  }

  Widget _pipelineStep(int number, String title, String description) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 22),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 28,
            height: 28,
            decoration: BoxDecoration(
              color: AppColors.primary.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Center(
              child: Text('$number',
                  style: const TextStyle(
                      color: AppColors.primary,
                      fontWeight: FontWeight.w700,
                      fontSize: 14)),
            ),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title,
                    style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: AppColors.textPrimary)),
                const SizedBox(height: 3),
                Text(description,
                    style: const TextStyle(
                        fontSize: 13.5,
                        color: AppColors.textSecondary,
                        height: 1.45)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Clean stat column for hero
class _StatColumn extends StatelessWidget {
  final String value;
  final String unit;
  final String label;

  const _StatColumn(
      {required this.value, required this.unit, required this.label});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            Text(value,
                style: const TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: Colors.white)),
            if (unit.isNotEmpty) ...[
              const SizedBox(width: 2),
              Padding(
                padding: const EdgeInsets.only(bottom: 2),
                child: Text(unit,
                    style: TextStyle(
                        fontSize: 11,
                        color: Colors.white.withValues(alpha: 0.7))),
              ),
            ],
          ],
        ),
        const SizedBox(height: 2),
        Text(label,
            style: TextStyle(
                fontSize: 10.5,
                color: Colors.white.withValues(alpha: 0.6))),
      ],
    );
  }
}

class _StatDivider extends StatelessWidget {
  const _StatDivider();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 1,
      height: 32,
      color: Colors.white.withValues(alpha: 0.15),
    );
  }
}

// Polished action card
class _ActionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;
  final String trailing;
  final VoidCallback onTap;

  const _ActionCard({
    required this.icon,
    required this.title,
    required this.description,
    required this.trailing,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: AppColors.primary.withValues(alpha: 0.07),
                  borderRadius: BorderRadius.circular(11),
                ),
                child: Icon(icon, color: AppColors.primary, size: 22),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title,
                        style: const TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.w600,
                            color: AppColors.textPrimary)),
                    const SizedBox(height: 2),
                    Text(description,
                        style: const TextStyle(
                            fontSize: 12.5,
                            color: AppColors.textSecondary)),
                  ],
                ),
              ),
              if (trailing.isNotEmpty)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: AppColors.surface,
                    borderRadius: BorderRadius.circular(6),
                  ),
                  child: Text(trailing,
                      style: const TextStyle(
                          fontSize: 11,
                          color: AppColors.textTertiary,
                          fontWeight: FontWeight.w500)),
                ),
              const SizedBox(width: 4),
              const Icon(Icons.chevron_right,
                  color: AppColors.textTertiary, size: 20),
            ],
          ),
        ),
      ),
    );
  }
}
