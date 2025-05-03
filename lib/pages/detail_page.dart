import 'package:flutter/material.dart';
import 'package:thermal_anomaly_detector/main.dart';
import 'package:thermal_anomaly_detector/utils/utilities.dart';

class DetailPage extends StatelessWidget {
  final AnalysisResult result;

  const DetailPage({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(result.hasAnomaly ? 'Anomaly Detail' : 'Normal Image'),
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(
              height: MediaQuery.of(context).size.height * 0.4, // reduced height
              child: Stack(
                children: [
                  Image.file(
                    result.image,
                    width: double.infinity,
                    fit: BoxFit.contain,
                  ),
                  if (result.hasAnomaly)
                    CustomPaint(
                      size: Size.infinite,
                      painter: BoundingBoxPainter(result.boundingBoxes),
                    ),
                ],
              ),
            ),
            const SizedBox(height: 8), // reduced spacing after image
            Padding(
              padding: const EdgeInsets.all(16.0),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            result.hasAnomaly
                                ? Icons.warning
                                : Icons.check_circle,
                            color: result.hasAnomaly ? Colors.red : Colors.green,
                            size: 24,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            result.hasAnomaly
                                ? 'Anomaly Detected'
                                : 'Normal Image',
                            style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      ResultInfoRow(
                        label: 'Confidence Score',
                        value:
                            '${(result.confidenceScore * 100).toStringAsFixed(1)}%',
                      ),
                      const Divider(),
                      ResultInfoRow(
                        label: 'Classification',
                        value: result.hasAnomaly
                            ? 'Potential Person'
                            : 'No Person',
                      ),
                      if (result.hasAnomaly) ...[
                        const Divider(),
                        ResultInfoRow(
                          label: 'Detected Regions',
                          value: '${result.boundingBoxes.length}',
                        ),
                      ],
                      const Divider(),
                      const ResultInfoRow(
                        label: 'Analysis Method',
                        value: 'PatchCore + Transfer Learning',
                      ),
                    ],
                  ),
                ),
              ),
            ),
            if (result.hasAnomaly)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: ElevatedButton.icon(
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                        content: Text('Alert sent to rescue team!'),
                        backgroundColor: Colors.green,
                      ),
                    );
                  },
                  icon: const Icon(Icons.send),
                  label: const Text('Alert Rescue Team'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}

class ResultInfoRow extends StatelessWidget {
  final String label;
  final String value;

  const ResultInfoRow({super.key, required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(fontSize: 14, color: Colors.grey[700]),
          ),
        ],
      ),
    );
  }
}