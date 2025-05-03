import 'dart:io';
import 'package:flutter/material.dart';
import 'package:thermal_anomaly_detector/pages/home_page.dart';

void main() {
  runApp(const AnomalyDetectionApp());
}

class AnomalyDetectionApp extends StatelessWidget {
  const AnomalyDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Thermal Anomaly Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        brightness:
            Brightness.dark, // Dark theme for better thermal image viewing
      ),
      home: const HomePage(),
    );
  }
}

class AnalysisResult {
  final File image;
  final bool hasAnomaly;
  final double confidenceScore;
  final List<Rect> boundingBoxes;

  AnalysisResult({
    required this.image,
    required this.hasAnomaly,
    required this.confidenceScore,
    required this.boundingBoxes,
  });
}