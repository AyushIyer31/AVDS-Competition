import 'dart:convert';
import 'package:http/http.dart' as http;
import '../models/api_models.dart';

class ApiService {
  // Change this to your backend URL
  // For local dev: http://10.0.2.2:8000 (Android emulator)
  // or http://localhost:8000 (iOS simulator)
  static const String baseUrl = 'http://10.0.0.76:8000';

  static Future<List<PDBResult>> searchPDB() async {
    final response = await http
        .get(Uri.parse('$baseUrl/pdb/search'))
        .timeout(const Duration(seconds: 30));

    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data
          .map((json) => PDBResult.fromJson(json as Map<String, dynamic>))
          .toList();
    }
    throw Exception('Failed to search PDB: ${response.statusCode}');
  }

  static Future<String> getDefaultSequence() async {
    final response = await http
        .get(Uri.parse('$baseUrl/default-sequence'))
        .timeout(const Duration(seconds: 10));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['sequence'] as String;
    }
    throw Exception('Failed to get default sequence');
  }

  static Future<List<BeneficialMutation>> scanMutations(
      String sequence) async {
    final response = await http
        .post(
          Uri.parse('$baseUrl/esm/mutations'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'sequence': sequence, 'name': 'query'}),
        )
        .timeout(const Duration(seconds: 120));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final mutations = data['beneficial_mutations'] as List;
      return mutations
          .map((m) => BeneficialMutation.fromJson(m as Map<String, dynamic>))
          .toList();
    }
    throw Exception('Failed to scan mutations: ${response.statusCode}');
  }

  static Future<OptimizationResult> optimize({
    required String sequence,
    int numCandidates = 10,
    int optimizationSteps = 50,
    double targetTemperature = 60.0,
  }) async {
    final response = await http
        .post(
          Uri.parse('$baseUrl/optimize'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'sequence': sequence,
            'num_candidates': numCandidates,
            'optimization_steps': optimizationSteps,
            'target_temperature': targetTemperature,
          }),
        )
        .timeout(const Duration(seconds: 300));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return OptimizationResult.fromJson(data as Map<String, dynamic>);
    }
    throw Exception('Optimization failed: ${response.statusCode}');
  }

  static Future<String?> lookupPdbId(String sequence) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/lookup-pdb?sequence=$sequence'))
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final pdbId = data['pdb_id'];
        return (pdbId != null && pdbId.toString().isNotEmpty) ? pdbId.toString() : null;
      }
    } catch (_) {}
    return null;
  }

  static Future<bool> checkHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
