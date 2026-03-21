import 'package:flutter/material.dart';
import 'theme.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const PETaseApp());
}

class PETaseApp extends StatelessWidget {
  const PETaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PETase ML Optimizer',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      home: const HomeScreen(),
    );
  }
}
