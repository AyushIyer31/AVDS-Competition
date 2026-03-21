import 'package:flutter/material.dart';
import '../theme.dart';
import '../services/api_service.dart';
import 'results_screen.dart';

class OptimizeScreen extends StatefulWidget {
  final String? initialSequence;

  const OptimizeScreen({super.key, this.initialSequence});

  @override
  State<OptimizeScreen> createState() => _OptimizeScreenState();
}

class _OptimizeScreenState extends State<OptimizeScreen> {
  final _sequenceController = TextEditingController();
  double _targetTemp = 60.0;
  int _numCandidates = 10;
  bool _loading = false;
  String? _error;
  String _statusMessage = '';
  int _step = 0;

  @override
  void initState() {
    super.initState();
    if (widget.initialSequence != null) {
      _sequenceController.text = widget.initialSequence!;
    }
  }

  @override
  void dispose() {
    _sequenceController.dispose();
    super.dispose();
  }

  Future<void> _loadDefault() async {
    try {
      final seq = await ApiService.getDefaultSequence();
      _sequenceController.text = seq;
    } catch (e) {
      _sequenceController.text =
          'MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAG'
          'TVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALR'
          'QVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTL'
          'IFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDT'
          'RYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ';
    }
  }

  Future<void> _runOptimization() async {
    final sequence =
        _sequenceController.text.trim().replaceAll(RegExp(r'\s+'), '');
    if (sequence.length < 10) {
      setState(
          () => _error = 'Sequence too short. Paste at least 10 amino acids.');
      return;
    }

    final validAA = RegExp(r'^[ACDEFGHIKLMNPQRSTVWY]+$');
    if (!validAA.hasMatch(sequence)) {
      setState(() => _error =
          'Invalid characters. Use only the 20 standard amino acid letters.');
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
      _step = 1;
      _statusMessage = 'Reading enzyme with ESM-2...';
    });

    try {
      setState(() {
        _step = 1;
        _statusMessage = 'Running ESM-2 optimization...';
      });

      final result = await ApiService.optimize(
        sequence: sequence,
        numCandidates: _numCandidates,
        targetTemperature: _targetTemp,
      );

      if (!mounted) return;

      Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => ResultsScreen(result: result)),
      );
    } catch (e) {
      setState(() =>
          _error = 'Could not reach ML server. Is the backend running?');
    } finally {
      if (mounted) {
        setState(() {
          _loading = false;
          _statusMessage = '';
          _step = 0;
        });
      }
    }
  }

  String _tempLabel() {
    if (_targetTemp <= 40) return 'Ambient — natural enzymes work here';
    if (_targetTemp <= 55) return 'Warm — enzymes start to struggle';
    if (_targetTemp <= 65) return 'PET glass transition — ideal for recycling';
    if (_targetTemp <= 75) return 'High heat — aggressive recycling';
    return 'Extreme — very challenging for enzymes';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(title: const Text('Design Better Enzyme')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Section 1: Sequence input
            const Text('ENZYME SEQUENCE',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 6),
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Each letter represents one amino acid building block',
                    style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
                  ),
                ),
                TextButton(
                  onPressed: _loading ? null : _loadDefault,
                  child: const Text('Use IsPETase',
                      style: TextStyle(fontSize: 13)),
                ),
              ],
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _sequenceController,
              maxLines: 5,
              style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 13,
                  color: AppColors.textPrimary),
              decoration: const InputDecoration(
                hintText: 'MNFPRASRLMQAAVLGGLMAVSAAATAQ...',
                hintStyle: TextStyle(color: AppColors.textTertiary),
              ),
            ),

            const SizedBox(height: 28),

            // Section 2: Temperature
            const Text('TARGET TEMPERATURE',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 4),
            const Text(
              'Higher temp speeds up recycling but enzymes denature. The AI optimizes for survival at your target.',
              style: TextStyle(fontSize: 13, color: AppColors.textSecondary, height: 1.4),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: Column(
                children: [
                  Row(
                    children: [
                      Text('${_targetTemp.round()}',
                          style: const TextStyle(
                              fontSize: 32,
                              fontWeight: FontWeight.w800,
                              color: AppColors.textPrimary)),
                      const Text(' °C',
                          style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w500,
                              color: AppColors.textSecondary)),
                      const Spacer(),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: _targetTemp >= 55 && _targetTemp <= 65
                              ? AppColors.success.withValues(alpha: 0.1)
                              : AppColors.surface,
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(_tempLabel(),
                            style: TextStyle(
                                fontSize: 11,
                                color: _targetTemp >= 55 && _targetTemp <= 65
                                    ? AppColors.success
                                    : AppColors.textTertiary,
                                fontWeight: FontWeight.w500)),
                      ),
                    ],
                  ),
                  SliderTheme(
                    data: SliderThemeData(
                      activeTrackColor: AppColors.primary,
                      inactiveTrackColor: AppColors.border,
                      thumbColor: AppColors.primary,
                      overlayColor: AppColors.primary.withValues(alpha: 0.1),
                      trackHeight: 4,
                    ),
                    child: Slider(
                      value: _targetTemp,
                      min: 30,
                      max: 90,
                      divisions: 12,
                      onChanged: _loading
                          ? null
                          : (v) => setState(() => _targetTemp = v),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 24),

            // Section 3: Candidates
            const Text('CANDIDATES TO GENERATE',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 4),
            const Text(
              'Each candidate is a unique enzyme variant the AI predicts will perform better.',
              style: TextStyle(fontSize: 13, color: AppColors.textSecondary, height: 1.4),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: Row(
                children: [
                  Text('$_numCandidates',
                      style: const TextStyle(
                          fontSize: 28,
                          fontWeight: FontWeight.w800,
                          color: AppColors.textPrimary)),
                  const SizedBox(width: 8),
                  const Text('variants',
                      style: TextStyle(
                          fontSize: 14, color: AppColors.textSecondary)),
                  const Spacer(),
                  SizedBox(
                    width: 180,
                    child: SliderTheme(
                      data: SliderThemeData(
                        activeTrackColor: AppColors.primary,
                        inactiveTrackColor: AppColors.border,
                        thumbColor: AppColors.primary,
                        overlayColor: AppColors.primary.withValues(alpha: 0.1),
                        trackHeight: 4,
                      ),
                      child: Slider(
                        value: _numCandidates.toDouble(),
                        min: 5,
                        max: 25,
                        divisions: 4,
                        onChanged: _loading
                            ? null
                            : (v) =>
                                setState(() => _numCandidates = v.round()),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Error
            if (_error != null) ...[
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppColors.error.withValues(alpha: 0.06),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                      color: AppColors.error.withValues(alpha: 0.2)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(Icons.error_outline,
                        color: AppColors.error, size: 18),
                    const SizedBox(width: 10),
                    Expanded(
                        child: Text(_error!,
                            style: const TextStyle(
                                color: AppColors.error,
                                fontSize: 13,
                                height: 1.3))),
                  ],
                ),
              ),
            ],

            const SizedBox(height: 28),

            // Run button
            SizedBox(
              width: double.infinity,
              height: 54,
              child: ElevatedButton(
                onPressed: _loading ? null : _runOptimization,
                child: _loading
                    ? Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.white),
                          ),
                          const SizedBox(width: 12),
                          Text('Step $_step/3: $_statusMessage',
                              style: const TextStyle(fontSize: 13)),
                        ],
                      )
                    : const Text('Run AI Optimization',
                        style: TextStyle(fontSize: 16)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
