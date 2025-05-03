import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:thermal_anomaly_detector/main.dart';
import 'package:thermal_anomaly_detector/pages/result_page.dart';

class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  _DetectionPageState createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  final ImagePicker _picker = ImagePicker();
  List<File> _selectedImages = [];
  List<AnalysisResult> _results = [];
  bool _isProcessing = false;
  bool _isAnalysisComplete = false;

  Future<void> _pickImagesFromGallery() async {
    try {
      final List<XFile> pickedFiles = await _picker.pickMultiImage();
      if (pickedFiles != null && pickedFiles.isNotEmpty) {
        setState(() {
          _selectedImages = pickedFiles.map((file) => File(file.path)).toList();
          _isAnalysisComplete = false;
          _results = [];
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error picking images: $e')),
      );
    }
  }

  Future<void> _pickFolder() async {
    try {
      String? selectedDirectory = await FilePicker.platform.getDirectoryPath();
      if (selectedDirectory != null) {
        final directory = Directory(selectedDirectory);
        final List<FileSystemEntity> entities = directory.listSync();
        final imageFiles = entities
            .whereType<File>()
            .where((file) =>
                file.path.endsWith('.jpg') ||
                file.path.endsWith('.jpeg') ||
                file.path.endsWith('.png'))
            .toList();

        setState(() {
          _selectedImages = imageFiles;
          _isAnalysisComplete = false;
          _results = [];
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error picking folder: $e')),
      );
    }
  }

  Future<void> _analyzeImages() async {
    if (_selectedImages.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select images first')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
    });

    // Simulate analysis with mock results
    await Future.delayed(const Duration(seconds: 3));

    // Generate mock analysis results
    final List<AnalysisResult> results = [];
    for (var i = 0; i < _selectedImages.length; i++) {
      final isAnomaly = i % 3 == 0; // Mock: every 3rd image has anomaly
      final confidenceScore =
          isAnomaly ? 0.7 + (i % 3) * 0.1 : 0.2 + (i % 5) * 0.05;

      results.add(AnalysisResult(
        image: _selectedImages[i],
        hasAnomaly: isAnomaly,
        confidenceScore: confidenceScore,
        boundingBoxes: isAnomaly
            ? [
                Rect.fromLTWH(100, 100, 50, 70),
              ]
            : [],
      ));
    }

    // Sort results by confidence score (highest first)
    results.sort((a, b) => b.confidenceScore.compareTo(a.confidenceScore));

    setState(() {
      _isProcessing = false;
      _isAnalysisComplete = true;
      _results = results;
    });

    // Navigate to results page
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultsPage(results: _results),
      ),
    );
  }

  // Function to delete an image from the selected list
  void _deleteImage(int index) {
    setState(() {
      _selectedImages.removeAt(index);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Detection Setup'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: ListView(
                children: [
                  Card(
                    elevation: 4,
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Import Images',
                            style: TextStyle(
                                fontSize: 18, fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 16),
                          Wrap(
                            spacing: 16,
                            runSpacing: 10,
                            alignment: WrapAlignment.center,
                            children: [
                              ElevatedButton.icon(
                                onPressed: _pickImagesFromGallery,
                                icon: const Icon(Icons.photo_library),
                                label: const Text('Select Images'),
                              ),
                              ElevatedButton.icon(
                                onPressed: _pickFolder,
                                icon: const Icon(Icons.folder),
                                label: const Text('Select Folder'),
                              ),
                            ],
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'Selected: ${_selectedImages.length} images',
                            style: const TextStyle(fontSize: 16),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),
                  if (_selectedImages.isNotEmpty) ...[
                    const Text(
                      'Preview:',
                      style:
                          TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    SizedBox(
                      height: 100,
                      child: ListView.builder(
                        scrollDirection: Axis.horizontal,
                        itemCount: _selectedImages.length > 10
                            ? 10
                            : _selectedImages.length,
                        itemBuilder: (context, index) {
                          return Padding(
                            padding: const EdgeInsets.only(right: 8.0),
                            child: Stack(
                              children: [
                                Image.file(
                                  _selectedImages[index],
                                  height: 80,
                                  width: 100,
                                  fit: BoxFit.cover,
                                ),
                                Positioned(
                                  bottom: 0,
                                  left: 0,
                                  child: IconButton(
                                    icon: const Icon(
                                      Icons.delete,
                                      color: Colors.red,
                                      size: 20,
                                    ),
                                    onPressed: () => _deleteImage(index),
                                  ),
                                ),
                              ],
                            ),
                          );
                        },
                      ),
                    ),
                    if (_selectedImages.length > 10)
                      Text('+ ${_selectedImages.length - 10} more images'),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isProcessing ? null : _analyzeImages,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 15),
                backgroundColor: Colors.green,
                minimumSize: const Size.fromHeight(50),
              ),
              child: _isProcessing
                  ? const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(color: Colors.white),
                        ),
                        SizedBox(width: 10),
                        Text('Processing...'),
                      ],
                    )
                  : const Text('Start Analysis'),
            ),
          ],
        ),
      ),
    );
  }
}
