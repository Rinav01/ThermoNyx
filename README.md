ThermoNyx: AI-Powered Search & Rescue Assistant
1. Challenge Tackled
Problem: Manual thermal image analysis delays missing person recovery. 
Users: Mountain rescue teams needing real-time anomaly detection in drone footage.

2. Tools/ML Models Used
- RandomForestClassifier: Core detection model (50 estimators) 
- OpenCV:
Feature extraction (8 thermal characteristics) 
- Streamlit:
Web dashboard visualization 
- Flutter:
Field deployment mobile app 

3. What Worked Well
- 85% confidence threshold reduced false positives
- Cross-platform design (web + mobile) enabled field use
- 8-feature extraction pipeline handled thermal variability 

4. Key Challenges
- Real-time processing: Added mock analysis for mobile testing [Flutter delay code]
- Thermal variability: Implemented brightness normalization
- Image sizing: Flask API 5MB limit with validation 
-Integration of app and model(as each functioning individually)

5. Development Timeline
- 0-4h:
Requirements analysis & dataset collection
- 4-8h:
Flask API + RandomForest implementation
- 8-14h:
Python dashboard + Flutter mobile app
- 14-20h:
System integration testing
- 20-24h:
Documentation & video preparation
Future Improvements
With 24 more hours: Implement real-time video processing pipeline and CNN model
complimenting RandomForest

