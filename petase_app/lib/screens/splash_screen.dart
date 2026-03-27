import 'dart:math';
import 'package:flutter/material.dart';
import 'home_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late AnimationController _helixController;
  late AnimationController _fadeController;
  late AnimationController _textController;
  late AnimationController _exitController;

  late Animation<double> _helixProgress;
  late Animation<double> _glowOpacity;
  late Animation<double> _textOpacity;
  late Animation<double> _textSlide;
  late Animation<double> _subtitleOpacity;
  late Animation<double> _particleFade;
  late Animation<double> _exitFade;

  @override
  void initState() {
    super.initState();

    // Helix draws in over 1.4s
    _helixController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    );
    _helixProgress = CurvedAnimation(
      parent: _helixController,
      curve: Curves.easeOutCubic,
    );
    _glowOpacity = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _helixController,
        curve: const Interval(0.5, 1.0, curve: Curves.easeIn),
      ),
    );

    // Text fades in after helix
    _textController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _textOpacity = CurvedAnimation(
      parent: _textController,
      curve: Curves.easeOut,
    );
    _textSlide = Tween<double>(begin: 20, end: 0).animate(
      CurvedAnimation(parent: _textController, curve: Curves.easeOutCubic),
    );
    _subtitleOpacity = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(
        parent: _textController,
        curve: const Interval(0.3, 1.0, curve: Curves.easeOut),
      ),
    );

    // Floating particles
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    );
    _particleFade = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeInOut,
    );

    // Exit transition
    _exitController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 500),
    );
    _exitFade = CurvedAnimation(
      parent: _exitController,
      curve: Curves.easeInCubic,
    );

    _startSequence();
  }

  Future<void> _startSequence() async {
    await Future.delayed(const Duration(milliseconds: 200));
    _helixController.forward();
    _fadeController.repeat(reverse: true);

    await Future.delayed(const Duration(milliseconds: 1000));
    _textController.forward();

    await Future.delayed(const Duration(milliseconds: 1800));
    _exitController.forward();

    await Future.delayed(const Duration(milliseconds: 500));
    if (mounted) {
      Navigator.of(context).pushReplacement(
        PageRouteBuilder(
          pageBuilder: (_, __, ___) => const HomeScreen(),
          transitionDuration: const Duration(milliseconds: 600),
          transitionsBuilder: (_, anim, __, child) {
            return FadeTransition(opacity: anim, child: child);
          },
        ),
      );
    }
  }

  @override
  void dispose() {
    _helixController.dispose();
    _fadeController.dispose();
    _textController.dispose();
    _exitController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      body: AnimatedBuilder(
        animation: Listenable.merge([
          _helixController,
          _textController,
          _fadeController,
          _exitController,
        ]),
        builder: (context, _) {
          return Opacity(
            opacity: 1.0 - _exitFade.value,
            child: Stack(
              children: [
                // Background gradient
                Container(
                  decoration: const BoxDecoration(
                    gradient: RadialGradient(
                      center: Alignment(0, -0.2),
                      radius: 1.2,
                      colors: [
                        Color(0xFF222244),
                        Color(0xFF1A1A2E),
                      ],
                    ),
                  ),
                ),

                // Floating particles
                ...List.generate(12, (i) => _buildParticle(i)),

                // Main content
                Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Double helix
                      SizedBox(
                        width: 200,
                        height: 280,
                        child: CustomPaint(
                          painter: _HelixPainter(
                            progress: _helixProgress.value,
                            glowOpacity: _glowOpacity.value,
                          ),
                        ),
                      ),

                      const SizedBox(height: 36),

                      // "PET" text
                      Transform.translate(
                        offset: Offset(0, _textSlide.value),
                        child: Opacity(
                          opacity: _textOpacity.value,
                          child: const Text(
                            'PET',
                            style: TextStyle(
                              fontSize: 52,
                              fontWeight: FontWeight.w800,
                              color: Color(0xFF1A7FC4),
                              letterSpacing: 6,
                              height: 1,
                            ),
                          ),
                        ),
                      ),

                      const SizedBox(height: 4),

                      // "LAB" text
                      Transform.translate(
                        offset: Offset(0, _textSlide.value * 0.6),
                        child: Opacity(
                          opacity: _subtitleOpacity.value,
                          child: const Text(
                            'LAB',
                            style: TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.w400,
                              color: Color(0xFF0FB5A2),
                              letterSpacing: 14,
                              height: 1,
                            ),
                          ),
                        ),
                      ),

                      const SizedBox(height: 24),

                      // Tagline
                      Opacity(
                        opacity: _subtitleOpacity.value * 0.7,
                        child: const Text(
                          'AI-Powered Enzyme Engineering',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w400,
                            color: Color(0xFF6B8AB5),
                            letterSpacing: 1.5,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildParticle(int index) {
    final rng = Random(index * 37);
    final startX = rng.nextDouble();
    final startY = rng.nextDouble();
    final size = 2.0 + rng.nextDouble() * 4;
    final isAqua = index % 2 == 0;
    final baseOpacity = 0.15 + rng.nextDouble() * 0.2;
    final phaseOffset = rng.nextDouble();

    return AnimatedBuilder(
      animation: _fadeController,
      builder: (context, _) {
        final screen = MediaQuery.of(context).size;
        final t = (_particleFade.value + phaseOffset) % 1.0;
        final opacity = baseOpacity * sin(t * pi);
        final drift = sin((_fadeController.value + phaseOffset) * pi * 2) * 8;

        return Positioned(
          left: startX * screen.width + drift,
          top: startY * screen.height - 20 + drift * 0.5,
          child: Opacity(
            opacity: opacity.clamp(0.0, 1.0),
            child: Container(
              width: size,
              height: size,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: isAqua
                    ? const Color(0xFF0FB5A2)
                    : const Color(0xFF1A7FC4),
              ),
            ),
          ),
        );
      },
    );
  }
}

// Animated double helix painter
class _HelixPainter extends CustomPainter {
  final double progress;
  final double glowOpacity;

  _HelixPainter({required this.progress, required this.glowOpacity});

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final top = 0.0;
    final bot = size.height;
    final amplitude = size.width * 0.32;
    final strandWidth = 7.0;

    final aqua = Color(0xFF0FB5A2);
    final blue = Color(0xFF1A7FC4);

    // How far to draw (animated)
    final drawEnd = progress;

    // Draw strands
    final aquaPath = Path();
    final bluePath = Path();
    final bondPaint = Paint()
      ..strokeWidth = 1.2
      ..style = PaintingStyle.stroke;

    final nSegments = 120;
    final bonds = <_Bond>[];

    for (int i = 0; i <= nSegments; i++) {
      final t = i / nSegments;
      if (t > drawEnd) break;

      final y = top + t * (bot - top);
      final wave = sin(t * pi * 3.0) * amplitude;

      final ax = cx + wave;
      final bx = cx - wave;

      if (i == 0) {
        aquaPath.moveTo(ax, y);
        bluePath.moveTo(bx, y);
      } else {
        aquaPath.lineTo(ax, y);
        bluePath.lineTo(bx, y);
      }

      // Hydrogen bonds at crossover points
      if (i % 20 == 10 && t <= drawEnd) {
        bonds.add(_Bond(ax, bx, y, t));
      }
    }

    // Active site glow
    final midY = size.height * 0.45;
    if (glowOpacity > 0) {
      final glowPaint = Paint()..style = PaintingStyle.fill;

      for (final r in [40.0, 28.0, 14.0]) {
        final alpha = (glowOpacity * (r == 14 ? 0.3 : r == 28 ? 0.15 : 0.08));
        glowPaint.color = aqua.withValues(alpha: alpha);
        canvas.drawCircle(Offset(cx, midY), r, glowPaint);
      }
    }

    // Draw bonds (behind strands)
    for (final bond in bonds) {
      final bondAlpha = ((bond.t / drawEnd) * 0.35).clamp(0.0, 0.35);
      bondPaint.color = (bond.t < 0.5 ? blue : aqua).withValues(alpha: bondAlpha);
      canvas.drawLine(
        Offset(bond.x1, bond.y),
        Offset(bond.x2, bond.y),
        bondPaint
          ..strokeWidth = 1.2
          ..style = PaintingStyle.stroke,
      );
    }

    // Draw aqua strand
    canvas.drawPath(
      aquaPath,
      Paint()
        ..color = aqua.withValues(alpha: 0.9)
        ..strokeWidth = strandWidth
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round,
    );

    // Draw blue strand
    canvas.drawPath(
      bluePath,
      Paint()
        ..color = blue.withValues(alpha: 0.9)
        ..strokeWidth = strandWidth
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round,
    );

    // Bright tip dots at the drawing edge
    if (progress < 1.0 && progress > 0.05) {
      final tipT = drawEnd;
      final tipY = top + tipT * (bot - top);
      final tipWave = sin(tipT * pi * 3.0) * amplitude;

      final tipPaint = Paint()..style = PaintingStyle.fill;

      tipPaint.color = aqua;
      canvas.drawCircle(Offset(cx + tipWave, tipY), 5, tipPaint);

      tipPaint.color = blue;
      canvas.drawCircle(Offset(cx - tipWave, tipY), 5, tipPaint);
    }
  }

  @override
  bool shouldRepaint(_HelixPainter old) =>
      old.progress != progress || old.glowOpacity != glowOpacity;
}

class _Bond {
  final double x1, x2, y, t;
  _Bond(this.x1, this.x2, this.y, this.t);
}

