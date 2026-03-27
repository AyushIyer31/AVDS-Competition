import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:http/http.dart' as http;
import '../theme.dart';
import '../services/api_service.dart';

class StructureViewerScreen extends StatefulWidget {
  final String originalSequence;
  final String candidateSequence;
  final List<String> mutations;
  final int rank;

  const StructureViewerScreen({
    super.key,
    required this.originalSequence,
    required this.candidateSequence,
    required this.mutations,
    required this.rank,
  });

  @override
  State<StructureViewerScreen> createState() => _StructureViewerScreenState();
}

class _StructureViewerScreenState extends State<StructureViewerScreen> {
  late final WebViewController _controller;
  bool _loading = true;
  String? _error;
  String _statusText = 'Predicting 3D structure with ESMFold...';

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFF1A1A2E))
      ..setNavigationDelegate(NavigationDelegate(
        onPageFinished: (_) {
          if (mounted) setState(() => _loading = false);
        },
        onWebResourceError: (error) {
          if (mounted) {
            setState(() {
              _loading = false;
              _error = 'Failed to load 3D viewer';
            });
          }
        },
      ));
    _loadStructure();
  }

  Future<void> _loadStructure() async {
    try {
      setState(() => _statusText = 'Generating 3D structure with AI...');

      // POST the candidate sequence to backend — it will use ESMFold or PDB
      final response = await http.post(
        Uri.parse('${ApiService.baseUrl}/api/structure-viewer'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'sequence': widget.candidateSequence,
          'original_sequence': widget.originalSequence,
          'mutations': widget.mutations.join(','),
          'title': 'Candidate #${widget.rank}',
        }),
      ).timeout(const Duration(seconds: 180));

      if (response.statusCode == 200) {
        // Load the returned HTML directly into the WebView
        _controller.loadHtmlString(response.body);
      } else {
        setState(() {
          _loading = false;
          _error = 'Structure prediction failed (${response.statusCode}). '
              'The ESMFold server may be busy — try again in a moment.';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _loading = false;
          _error = 'Could not generate structure: $e';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1A2E),
        foregroundColor: Colors.white,
        title: Text(
          'Candidate #${widget.rank} — 3D Structure',
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
      body: _error != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.view_in_ar, size: 48, color: Colors.white24),
                    const SizedBox(height: 16),
                    Text(
                      _error!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(color: Colors.white60, fontSize: 14, height: 1.5),
                    ),
                    const SizedBox(height: 24),
                    TextButton(
                      onPressed: () {
                        setState(() {
                          _error = null;
                          _loading = true;
                        });
                        _loadStructure();
                      },
                      child: const Text('Retry', style: TextStyle(color: AppColors.primary)),
                    ),
                  ],
                ),
              ),
            )
          : Stack(
              children: [
                WebViewWidget(controller: _controller),
                if (_loading)
                  Container(
                    color: const Color(0xFF1A1A2E),
                    child: Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const CircularProgressIndicator(
                            color: AppColors.primary,
                            strokeWidth: 2.5,
                          ),
                          const SizedBox(height: 20),
                          Text(
                            _statusText,
                            style: const TextStyle(color: Colors.white60, fontSize: 14),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            '${widget.mutations.length} mutation(s): ${widget.mutations.join(", ")}',
                            style: const TextStyle(color: Colors.white38, fontSize: 12),
                          ),
                          const SizedBox(height: 6),
                          const Text(
                            'This may take 15-30 seconds',
                            style: TextStyle(color: Colors.white24, fontSize: 11),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
    );
  }
}
