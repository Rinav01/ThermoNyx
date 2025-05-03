import 'package:flutter/material.dart';
import 'package:thermal_anomaly_detector/main.dart';
import 'package:thermal_anomaly_detector/pages/detail_page.dart';

class ResultsPage extends StatelessWidget {
  final List<AnalysisResult> results;

  const ResultsPage({Key? key, required this.results}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Count anomalies
    final anomalyCount = results.where((result) => result.hasAnomaly).length;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Analysis Results'),
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Card(
              color: Colors.blue.shade900,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Text(
                      'Analysis Complete',
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Analyzed ${results.length} images',
                      style: const TextStyle(fontSize: 16),
                    ),
                    Text(
                      'Found $anomalyCount potential anomalies',
                      style: TextStyle(
                        fontSize: 16,
                        color: anomalyCount > 0 ? Colors.red : Colors.green,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: results.length,
              itemBuilder: (context, index) {
                final result = results[index];
                return Card(
                  margin:
                      const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  color: result.hasAnomaly
                      ? Colors.red.withOpacity(0.1)
                      : Colors.green.withOpacity(0.1),
                  child: ListTile(
                    leading: Image.file(
                      result.image,
                      width: 60,
                      height: 60,
                      fit: BoxFit.cover,
                    ),
                    title: Text(
                      result.hasAnomaly
                          ? 'Potential Person Detected'
                          : 'Normal Image',
                      style: TextStyle(
                        color: result.hasAnomaly ? Colors.red : Colors.green,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    subtitle: Text(
                      'Confidence: ${(result.confidenceScore * 100).toStringAsFixed(1)}%',
                    ),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => DetailPage(result: result),
                        ),
                      );
                    },
                  ),
                );
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton(
              onPressed: () {
                Navigator.popUntil(context, (route) => route.isFirst);
              },
              child: const Text('Back to Home'),
            ),
          ),
        ],
      ),
    );
  }
}