import 'package:flutter/material.dart';
import '../theme.dart';
import '../services/api_service.dart';
import 'results_screen.dart';

class _EnzymePreset {
  final String name;
  final String organism;
  final String pdbId;
  final String description;
  final int residues;
  final String sequence;

  const _EnzymePreset({
    required this.name,
    required this.organism,
    required this.pdbId,
    required this.description,
    required this.residues,
    required this.sequence,
  });
}

const _presets = [
  _EnzymePreset(
    name: 'IsPETase',
    organism: 'I. sakaiensis',
    pdbId: '5XJH',
    description: 'Wild-type, denatures at 40\u00b0C',
    residues: 312,
    sequence:
        'MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAG'
        'TVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALR'
        'QVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTL'
        'IFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDT'
        'RYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ',
  ),
  _EnzymePreset(
    name: 'ThermoPETase',
    organism: 'Engineered',
    pdbId: '6IJ6',
    description: '3 mutations, active at 72\u00b0C',
    residues: 312,
    sequence:
        'MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAG'
        'TVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPESRSSQQMAALR'
        'QVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWHSSTNFSSVTVPTL'
        'IFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDT'
        'RYSTFACENPNSTAVSDFRTANCSLEDPAANKARKEAELAAATAEQ',
  ),
  _EnzymePreset(
    name: 'FAST-PETase',
    organism: 'Engineered',
    pdbId: 'Lu et al. 2022',
    description: '5 mutations, full PET in 1 week',
    residues: 312,
    sequence:
        'MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAG'
        'TVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPESRSSQQMAALR'
        'QVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWHSSTNFSSVTVPTL'
        'IFACENDSIAPVNSSALPIYDSMSRNAKQFLEIKGGSHFCANSGNSNQALIGKKGVAWMKRFMDNDT'
        'RYSTFACENPNSTAVSDFRTANCSLEDPAANKARKEAELAAATAEQ',
  ),
  _EnzymePreset(
    name: 'LCC',
    organism: 'Metagenome',
    pdbId: '4EB0',
    description: 'Leaf-branch compost cutinase',
    residues: 261,
    sequence:
        'SNPYQRGPNPTRSALTADGPFSVATYTVSRLSVSGFGGGVIYYPTGTSLTFGGIAMSPGYTADASSL'
        'AWLGRRLASHGFVVLVINTNSRFDYPDSRASQLSAALNYLRTSSPSAVRARLDANRLAVAGHSMGGG'
        'GTLRIAEQNPSLKAAVPLTPWHTDKTFNTSVPVLIVGAEADTVAPVSQHAIPFYQNLPSTTPKVYV'
        'ELDNASHFAPNSNNAAISVYTISWMKLWVDNDTRYRQFLCNVNDPALSDFRTNNRHCQ',
  ),
];

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
  String? _selectedPreset;

  @override
  void initState() {
    super.initState();
    if (widget.initialSequence != null) {
      _sequenceController.text = widget.initialSequence!;
    }
    _sequenceController.addListener(() => setState(() {}));
  }

  @override
  void dispose() {
    _sequenceController.dispose();
    super.dispose();
  }

  void _selectPreset(_EnzymePreset preset) {
    _sequenceController.text = preset.sequence;
    setState(() => _selectedPreset = preset.name);
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
    if (_targetTemp <= 40) return 'Ambient \u2014 natural enzymes work here';
    if (_targetTemp <= 55) return 'Warm \u2014 enzymes start to struggle';
    if (_targetTemp <= 65) return 'PET glass transition \u2014 ideal for recycling';
    if (_targetTemp <= 75) return 'High heat \u2014 aggressive recycling';
    return 'Extreme \u2014 very challenging for enzymes';
  }

  @override
  Widget build(BuildContext context) {
    final hasSequence = _sequenceController.text.trim().isNotEmpty;
    final seqLength = _sequenceController.text.trim().replaceAll(RegExp(r'\s+'), '').length;

    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(title: const Text('Design Better Enzyme')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Section 1: Preset enzyme chips (only when no sequence was pre-loaded)
            if (widget.initialSequence == null) ...[
            const Text('SELECT AN ENZYME',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 4),
            const Text(
              'Pick a known enzyme or paste your own sequence below',
              style: TextStyle(fontSize: 13, color: AppColors.textSecondary),
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _presets.map((preset) {
                final isSelected = _selectedPreset == preset.name;
                return GestureDetector(
                  onTap: _loading ? null : () => _selectPreset(preset),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                    decoration: BoxDecoration(
                      color: isSelected ? AppColors.primary : Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: isSelected ? AppColors.primary : AppColors.border,
                        width: isSelected ? 1.5 : 1,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(preset.name,
                                style: TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w700,
                                    color: isSelected ? Colors.white : AppColors.textPrimary)),
                            const SizedBox(width: 6),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                              decoration: BoxDecoration(
                                color: isSelected
                                    ? Colors.white.withValues(alpha: 0.2)
                                    : AppColors.surface,
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: Text('${preset.residues} aa',
                                  style: TextStyle(
                                      fontSize: 10,
                                      fontWeight: FontWeight.w500,
                                      color: isSelected ? Colors.white70 : AppColors.textTertiary)),
                            ),
                          ],
                        ),
                        const SizedBox(height: 2),
                        Text(preset.description,
                            style: TextStyle(
                                fontSize: 11,
                                color: isSelected ? Colors.white70 : AppColors.textTertiary)),
                      ],
                    ),
                  ),
                );
              }).toList(),
            ),

            const SizedBox(height: 16),
            ], // end if (widget.initialSequence == null)

            // Sequence text field
            Row(
              children: [
                const Text('SEQUENCE',
                    style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textTertiary,
                        letterSpacing: 1.2)),
                if (hasSequence) ...[
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: AppColors.success.withValues(alpha: 0.1),
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text('$seqLength residues',
                        style: const TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            color: AppColors.success)),
                  ),
                ],
                const Spacer(),
                if (hasSequence)
                  GestureDetector(
                    onTap: _loading
                        ? null
                        : () {
                            _sequenceController.clear();
                            setState(() => _selectedPreset = null);
                          },
                    child: const Text('Clear',
                        style: TextStyle(fontSize: 12, color: AppColors.primary, fontWeight: FontWeight.w500)),
                  ),
              ],
            ),
            const SizedBox(height: 6),
            TextField(
              controller: _sequenceController,
              maxLines: 4,
              style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 12,
                  color: AppColors.textPrimary),
              decoration: const InputDecoration(
                hintText: 'Select a preset above or paste a sequence...',
                hintStyle: TextStyle(color: AppColors.textTertiary, fontSize: 12),
              ),
            ),

            const SizedBox(height: 24),

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
                      const Text(' \u00b0C',
                          style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w500,
                              color: AppColors.textSecondary)),
                      const Spacer(),
                      Flexible(
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: _targetTemp >= 55 && _targetTemp <= 65
                                ? AppColors.success.withValues(alpha: 0.1)
                                : AppColors.surface,
                            borderRadius: BorderRadius.circular(6),
                          ),
                          child: Text(_tempLabel(),
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                  fontSize: 11,
                                  color: _targetTemp >= 55 && _targetTemp <= 65
                                      ? AppColors.success
                                      : AppColors.textTertiary,
                                  fontWeight: FontWeight.w500)),
                        ),
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
