import 'package:flutter/material.dart';
import 'theme.dart';
import 'screens/splash_screen.dart';

void main() {
  runApp(const PETLabApp());
}

class PETLabApp extends StatelessWidget {
  const PETLabApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PET Lab',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light,
      home: const SplashScreen(),
    );
  }
}
