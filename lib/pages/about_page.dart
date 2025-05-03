import 'package:flutter/material.dart';

class AboutPage extends StatelessWidget {
  const AboutPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('About'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: const [
            Text(
              'Thermal Anomaly Detection',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 16),
            Text(
              'This application helps search and rescue teams identify potential missing persons in thermal drone footage using AI-powered anomaly detection.',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 16),
            Text(
              'How it works:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text(
              '1. Upload thermal drone images or folders\n'
              '2. Our AI model analyzes each image\n'
              '3. Potential human signatures are highlighted\n'
              '4. Results are prioritized by confidence\n'
              '5. Rescue teams can be alerted about findings',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 16),
            Text(
              'Technology:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 8),
            Text(
              'This app uses transfer learning techniques adapted from industrial anomaly detection models like PatchCore, repurposed for humanitarian search and rescue operations.',
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}