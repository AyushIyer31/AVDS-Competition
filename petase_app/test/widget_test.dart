import 'package:flutter_test/flutter_test.dart';
import 'package:petase_app/main.dart';

void main() {
  testWidgets('App renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const PETaseApp());
    expect(find.text('PETase\nML Optimizer'), findsOneWidget);
  });
}
