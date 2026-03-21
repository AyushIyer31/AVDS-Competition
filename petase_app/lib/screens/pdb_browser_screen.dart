import 'package:flutter/material.dart';
import '../theme.dart';
import '../models/api_models.dart';
import '../services/api_service.dart';
import 'optimize_screen.dart';

class PDBBrowserScreen extends StatefulWidget {
  const PDBBrowserScreen({super.key});

  @override
  State<PDBBrowserScreen> createState() => _PDBBrowserScreenState();
}

class _PDBBrowserScreenState extends State<PDBBrowserScreen> {
  List<PDBResult>? _results;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final results = await ApiService.searchPDB();
      setState(() {
        _results = results;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        title: const Text('Enzyme Database'),
        actions: [
          if (_results != null)
            Center(
              child: Padding(
                padding: const EdgeInsets.only(right: 16),
                child: Text('${_results!.length} structures',
                    style: const TextStyle(
                        fontSize: 13, color: AppColors.textTertiary)),
              ),
            ),
        ],
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_loading) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 32,
              height: 32,
              child: CircularProgressIndicator(
                  strokeWidth: 2.5, color: AppColors.primary),
            ),
            SizedBox(height: 16),
            Text('Fetching from RCSB Protein Data Bank...',
                style: TextStyle(color: AppColors.textSecondary, fontSize: 14)),
          ],
        ),
      );
    }

    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.cloud_off, size: 40, color: AppColors.textTertiary),
              const SizedBox(height: 16),
              const Text('Could not reach RCSB',
                  style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: AppColors.textPrimary)),
              const SizedBox(height: 6),
              Text('Check your internet connection and try again.',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 13, color: AppColors.textSecondary)),
              const SizedBox(height: 20),
              ElevatedButton(onPressed: _loadData, child: const Text('Retry')),
            ],
          ),
        ),
      );
    }

    if (_results == null || _results!.isEmpty) {
      return const Center(
          child: Text('No structures found',
              style: TextStyle(color: AppColors.textSecondary)));
    }

    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      itemCount: _results!.length,
      separatorBuilder: (_, __) => const SizedBox(height: 8),
      itemBuilder: (context, index) => _buildPDBCard(_results![index]),
    );
  }

  Widget _buildPDBCard(PDBResult result) {
    return Material(
      color: Colors.white,
      borderRadius: BorderRadius.circular(14),
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: () => _showDetail(result),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border),
          ),
          child: Row(
            children: [
              // PDB ID badge
              Container(
                width: 54,
                height: 54,
                decoration: BoxDecoration(
                  color: AppColors.primary.withValues(alpha: 0.07),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Center(
                  child: Text(
                    result.pdbId,
                    style: const TextStyle(
                      fontWeight: FontWeight.w800,
                      color: AppColors.primary,
                      fontSize: 13,
                      letterSpacing: -0.3,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      result.title,
                      style: const TextStyle(
                          fontSize: 13.5,
                          fontWeight: FontWeight.w500,
                          color: AppColors.textPrimary,
                          height: 1.3),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        Text(
                          result.organism,
                          style: const TextStyle(
                              fontSize: 12,
                              color: AppColors.textTertiary,
                              fontStyle: FontStyle.italic),
                        ),
                        if (result.resolution != null) ...[
                          const SizedBox(width: 8),
                          Container(
                            width: 3,
                            height: 3,
                            decoration: const BoxDecoration(
                                color: AppColors.textTertiary,
                                shape: BoxShape.circle),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            '${result.resolution!.toStringAsFixed(1)} A',
                            style: const TextStyle(
                                fontSize: 12, color: AppColors.textTertiary),
                          ),
                        ],
                        const SizedBox(width: 8),
                        Container(
                          width: 3,
                          height: 3,
                          decoration: const BoxDecoration(
                              color: AppColors.textTertiary,
                              shape: BoxShape.circle),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          '${result.sequence.length} aa',
                          style: const TextStyle(
                              fontSize: 12, color: AppColors.textTertiary),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right,
                  size: 20, color: AppColors.textTertiary),
            ],
          ),
        ),
      ),
    );
  }

  void _showDetail(PDBResult result) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.65,
        minChildSize: 0.3,
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
            const SizedBox(height: 20),
            Row(
              children: [
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: AppColors.primary.withValues(alpha: 0.07),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    result.pdbId,
                    style: const TextStyle(
                      fontWeight: FontWeight.w800,
                      color: AppColors.primary,
                      fontSize: 18,
                    ),
                  ),
                ),
                const Spacer(),
                if (result.resolution != null)
                  Text('${result.resolution!.toStringAsFixed(1)} A resolution',
                      style: const TextStyle(
                          fontSize: 13, color: AppColors.textTertiary)),
              ],
            ),
            const SizedBox(height: 12),
            Text(result.title,
                style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                    height: 1.3)),
            const SizedBox(height: 6),
            Text(result.organism,
                style: const TextStyle(
                    fontStyle: FontStyle.italic,
                    color: AppColors.textTertiary,
                    fontSize: 14)),
            const SizedBox(height: 20),
            const Text('SEQUENCE',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textTertiary,
                    letterSpacing: 1.2)),
            const SizedBox(height: 6),
            Text('${result.sequence.length} amino acids',
                style: const TextStyle(
                    fontSize: 12, color: AppColors.textSecondary)),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: AppColors.border),
              ),
              child: SelectableText(
                result.sequence,
                style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 12,
                    height: 1.6,
                    color: AppColors.textPrimary),
              ),
            ),
            const SizedBox(height: 24),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.pop(context);
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (_) =>
                          OptimizeScreen(initialSequence: result.sequence),
                    ),
                  );
                },
                child: const Text('Optimize This Enzyme'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
