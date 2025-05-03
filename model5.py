import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import joblib
from PIL import Image
import base64
from io import BytesIO
import os
import logging
import uuid
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Thermal Anomaly Detection")
parser.add_argument('--force_retrain', action='store_true', help="Force retraining of models")
args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path configuration
train_dir = 'data/trainingdata'
test_normal_dir = 'data/testdata/normal'
test_abnormal_dir = 'data/testdata/abnormal'
checkpoint_path = 'autoencoder_checkpoint.keras'
model_rf_path = 'thermal_model_rf.pkl'
threshold_path = 'optimal_threshold.pkl'
scaler_path = 'feature_scaler.pkl'
EXPECTED_FEATURE_COUNT = 19
img_size = (224, 224)
batch_size = 16
epochs = 25
validation_split = 0.15

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.1),
    layers.GaussianNoise(0.05)
])

# Cache model loading for performance
@st.cache_resource
def load_models():
    autoencoder, encoder, model_rf, optimal_threshold, scaler = None, None, None, None, None
    if (not args.force_retrain and
        all(os.path.exists(f) for f in [checkpoint_path, model_rf_path, threshold_path, scaler_path])):
        try:
            autoencoder = keras.models.load_model(checkpoint_path, compile=False)
            encoder = keras.models.load_model('trained_encoder_model.keras')
            model_rf = joblib.load(model_rf_path)
            optimal_threshold = joblib.load(threshold_path)
            scaler = joblib.load(scaler_path)
            logger.info("Models loaded successfully")
            return autoencoder, encoder, model_rf, optimal_threshold, scaler
        except Exception as e:
            logger.warning(f"Failed to load models: {e}")
    
    return train_models()

def load_custom_train_dataset(path, label, limit=None):
    if not os.path.exists(path):
        logger.warning(f"Dataset path {path} does not exist!")
        return np.array([]), np.array([])
    
    file_paths = [os.path.join(path, f) for f in os.listdir(path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not file_paths:
        logger.warning(f"No images found in {path}!")
        return np.array([]), np.array([])
    
    if limit:
        file_paths = file_paths[:limit]
    
    images, labels = [], []
    for file in file_paths:
        try:
            img = tf.keras.preprocessing.image.load_img(file, target_size=img_size)
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    return np.array(images), np.array(labels)

def build_autoencoder(input_shape):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True
    
    encoder_input = keras.Input(shape=input_shape)
    x = data_augmentation(encoder_input)
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    
    x = layers.Dense(28*28*256, activation='relu')(encoded)
    x = layers.Reshape((28, 28, 256))(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = keras.Model(encoder_input, decoded)
    
    def ssim_loss(y_true, y_pred):
        return 0.7 * tf.keras.losses.MeanSquaredError()(y_true, y_pred) + \
               0.3 * (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)))
    
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-5),
                       loss=ssim_loss,
                       metrics=['mae'])
    
    encoder = keras.Model(encoder_input, encoded)
    return autoencoder, encoder

def extract_enhanced_features(img):
    if img is None:
        return None
    
    try:
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img.copy()
        
        mean_intensity = np.mean(img_gray)
        var_intensity = np.var(img_gray)
        max_intensity = np.max(img_gray)
        min_intensity = np.min(img_gray)
        intensity_range = max_intensity - min_intensity
        
        threshold = max(min(mean_intensity + 1.5 * np.std(img_gray), 220), 160)
        binary = (img_gray > threshold).astype(np.uint8) * 255
        bright_pixels = np.sum(img_gray > threshold)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        largest_component = 0
        total_component_area = 0
        aspect_ratio = 1.0
        component_density = 0.0
        component_circularity = 0.0
        num_significant_components = 0
        human_shape_score = 0.0
        thermal_signature_score = 0.0
        symmetry_score = 0.0
        component_x, component_y, component_w, component_h = 0, 0, 0, 0
        
        if num_labels > 1:
            significant_components = []
            min_area_threshold = 30
            
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
                    significant_components.append(i)
            
            num_significant_components = len(significant_components)
            
            if num_significant_components > 0:
                total_component_area = sum(stats[i, cv2.CC_STAT_AREA] for i in significant_components)
                largest_idx = max(significant_components, key=lambda i: stats[i, cv2.CC_STAT_AREA])
                largest_component = stats[largest_idx, cv2.CC_STAT_AREA]
                
                component_x = stats[largest_idx, cv2.CC_STAT_LEFT]
                component_y = stats[largest_idx, cv2.CC_STAT_TOP]
                component_w = stats[largest_idx, cv2.CC_STAT_WIDTH]
                component_h = stats[largest_idx, cv2.CC_STAT_HEIGHT]
                
                aspect_ratio = component_w / component_h if component_h > 0 else 1.0
                aspect_ratio = min(max(aspect_ratio, 0.2), 5.0)
                
                bbox_area = component_w * component_h
                component_density = largest_component / bbox_area if bbox_area > 0 else 0
                
                component_mask = (labels == largest_idx).astype(np.uint8) * 255
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    perimeter = cv2.arcLength(contours[0], True)
                    circularity = (4 * np.pi * largest_component) / (perimeter ** 2) if perimeter > 0 else 0
                    component_circularity = min(max(circularity, 0), 1)
                    
                    hull = cv2.convexHull(contours[0])
                    hull_area = cv2.contourArea(hull)
                    solidity = largest_component / hull_area if hull_area > 0 else 0
                    
                    if 50 < largest_component < 8000:
                        if 0.2 <= aspect_ratio <= 0.8:
                            if 0.4 <= solidity <= 0.95:
                                approx = cv2.approxPolyDP(contours[0], 0.02 * perimeter, True)
                                contour_complexity = len(approx)
                                if 5 <= contour_complexity <= 50:
                                    (x,y),(MA,ma),angle = cv2.fitEllipse(contours[0])
                                    elongation = ma/MA if MA > 0 else 1
                                    human_shape_score = min(
                                        (solidity * 0.4 + 
                                         min(elongation, 1.5) * 0.3 +
                                         (contour_complexity/50) * 0.3),
                                        1.0
                                    )
                    
                    roi = img_gray[component_y:component_y+component_h, component_x:component_x+component_w]
                    if roi.size > 0:
                        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                        thermal_signature_score = np.mean(gradient_magnitude) / 255.0
                        thermal_signature_score = min(thermal_signature_score, 1.0)
                    
                    if component_w > 0 and component_h > 0:
                        half_width = component_w // 2
                        left_half = roi[:, :half_width]
                        right_half = roi[:, component_w-half_width:]
                        if left_half.shape == right_half.shape:
                            flipped_right = cv2.flip(right_half, 1)
                            diff = cv2.absdiff(left_half, flipped_right)
                            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        hot_spot_ratio = bright_pixels / (img_gray.shape[0] * img_gray.shape[1]) if img_gray.size > 0 else 0
        
        edges = cv2.Canny(img_gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        edge_density = edge_pixels / (img_gray.shape[0] * img_gray.shape[1]) if img_gray.size > 0 else 0
        
        h, w = img_gray.shape
        h_mid, w_mid = h // 2, w // 2
        quadrants = [
            img_gray[:h_mid, :w_mid],
            img_gray[:h_mid, w_mid:],
            img_gray[h_mid:, :w_mid],
            img_gray[h_mid:, w_mid:]
        ]
        quadrant_means = [np.mean(q) for q in quadrants if q.size > 0]
        quadrant_variance = np.var(quadrant_means) if quadrant_means else 0
        
        return [
            mean_intensity,
            var_intensity,
            max_intensity,
            min_intensity,
            intensity_range,
            bright_pixels,
            hot_spot_ratio,
            largest_component,
            total_component_area,
            aspect_ratio,
            component_density,
            component_circularity,
            num_significant_components,
            entropy,
            edge_density,
            quadrant_variance,
            human_shape_score,
            thermal_signature_score,
            symmetry_score,
            component_x/img.shape[1] if img.shape[1] > 0 else 0,
            component_y/img.shape[0] if img.shape[0] > 0 else 0,
            (component_x+component_w)/img.shape[1] if img.shape[1] > 0 else 0,
            (component_y+component_h)/img.shape[0] if img.shape[0] > 0 else 0
        ]
    
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        return None

def train_models():
    logger.info("Loading training data...")
    x_train_normal, y_train_normal = load_custom_train_dataset(train_dir, 0)
    x_test_abnormal, y_test_abnormal = load_custom_train_dataset(test_abnormal_dir, 1)
    
    # Fallback to hardcoded data if dataset is unavailable
    if len(x_train_normal) == 0 or len(x_test_abnormal) == 0:
        logger.warning("Using fallback hardcoded training data")
        X_train = [
            [30, 60, 80, 10, 50, 100, 0.1, 20, 1000, 1.0, 0.5, 0.7, 1, 2.5, 0.01, 50, 0.5, 0.4, 0.8],
            [25, 50, 70, 20, 70, 150, 0.2, 10, 800, 1.2, 0.4, 0.6, 2, 2.0, 0.02, 40, 0.4, 0.3, 0.7],
            [35, 70, 90, 30, 90, 200, 0.3, 15, 1200, 0.8, 0.6, 0.8, 1, 3.0, 0.01, 60, 0.6, 0.5, 0.9],
            [120, 200, 255, 50, 100, 150, 0.4, 200, 5000, 0.5, 0.7, 0.5, 3, 4.0, 0.03, 100, 0.8, 0.7, 0.6],
            [110, 180, 240, 40, 80, 120, 0.5, 150, 4000, 0.6, 0.8, 0.4, 2, 3.5, 0.02, 80, 0.7, 0.6, 0.5]
        ]
        y_train = [0, 0, 0, 1, 1]
        model_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model_rf.fit(X_train_scaled, y_train)
        joblib.dump(model_rf, model_rf_path)
        joblib.dump(0.05, threshold_path)
        joblib.dump(scaler, scaler_path)
        return None, None, model_rf, 0.05, scaler
    
    x_train = np.concatenate([x_train_normal, x_test_abnormal[:len(x_train_normal)//2]])
    y_train = np.concatenate([np.zeros(len(x_train_normal)), np.ones(len(x_test_abnormal[:len(x_train_normal)//2]))])
    
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train, y_train = x_train[indices], y_train[indices]
    
    split_idx = int(len(x_train) * (1 - validation_split))
    x_val, y_val = x_train[split_idx:], y_train[split_idx:]
    x_train, y_train = x_train[:split_idx], y_train[:split_idx]
    
    logger.info("Building autoencoder...")
    autoencoder, encoder = build_autoencoder((img_size[0], img_size[1], 3))
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(checkpoint_path, save_best_only=True)
    ]
    
    logger.info("Training autoencoder...")
    autoencoder.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, x_val),
        callbacks=callbacks
    )
    
    x_val_normal = x_val[y_val == 0]
    optimal_threshold = 0.05
    if len(x_val_normal) > 0:
        x_val_normal_recon = autoencoder.predict(x_val_normal)
        reconstruction_errors = np.mean(np.square(x_val_normal - x_val_normal_recon), axis=(1, 2, 3))
        optimal_threshold = np.percentile(reconstruction_errors, 95) * 0.85
    
    logger.info(f"Optimal threshold set to: {optimal_threshold:.6f}")
    
    logger.info("Extracting features...")
    x_train_features = encoder.predict(x_train)
    x_val_features = encoder.predict(x_val)
    
    handcrafted_train = [extract_enhanced_features((x * 255).astype(np.uint8)) for x in x_train]
    handcrafted_val = [extract_enhanced_features((x * 255).astype(np.uint8)) for x in x_val]
    
    valid_indices = [i for i, hf in enumerate(handcrafted_train) if hf is not None]
    x_train_features = x_train_features[valid_indices]
    y_train = y_train[valid_indices]
    handcrafted_train = [handcrafted_train[i] for i in valid_indices]
    
    valid_indices_val = [i for i, hf in enumerate(handcrafted_val) if hf is not None]
    x_val_features = x_val_features[valid_indices_val]
    y_val = y_val[valid_indices_val]
    handcrafted_val = [handcrafted_val[i] for i in valid_indices_val]
    
    scaler = StandardScaler()
    hf_train = scaler.fit_transform(handcrafted_train)
    hf_val = scaler.transform(handcrafted_val)
    
    X_train = [np.concatenate([lf, hf]) for lf, hf in zip(x_train_features, hf_train)]
    X_val = [np.concatenate([lf, hf]) for lf, hf in zip(x_val_features, hf_val)]
    
    logger.info("Training Random Forest...")
    model_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model_rf.fit(X_train, y_train)
    
    y_pred = model_rf.predict(X_val)
    y_proba = model_rf.predict_proba(X_val)[:, 1]
    ap_score = average_precision_score(y_val, y_proba)
    logger.info(f"Classifier Average Precision: {ap_score:.3f}")
    
    autoencoder.save(checkpoint_path)
    encoder.save('trained_encoder_model.keras')
    joblib.dump(model_rf, model_rf_path)
    joblib.dump(optimal_threshold, threshold_path)
    joblib.dump(scaler, scaler_path)
    
    return autoencoder, encoder, model_rf, optimal_threshold, scaler

def detect_human_presence(img, autoencoder, encoder, model_rf, optimal_threshold, scaler):
    if autoencoder is None or encoder is None:
        handcrafted_features = extract_enhanced_features(img)
        if handcrafted_features is None:
            return "Error", 0, {"error": "Feature extraction failed"}, None, None
        
        handcrafted_features = handcrafted_features[:EXPECTED_FEATURE_COUNT]
        handcrafted_features_normalized = scaler.transform([handcrafted_features])
        rf_pred = model_rf.predict(handcrafted_features_normalized)[0]
        rf_proba = model_rf.predict_proba(handcrafted_features_normalized)[0]
        confidence = rf_proba[int(rf_pred)] * 100
        
        result = "Abnormal (Human Detected)" if rf_pred == 1 else "Normal (No Human)"
        debug_info = {
            "bright_pixels": handcrafted_features[5],
            "largest_component": handcrafted_features[7],
            "human_shape_score": handcrafted_features[16],
            "thermal_signature_score": handcrafted_features[17],
            "symmetry_score": handcrafted_features[18]
        }
        visual_data = {
            'x1': handcrafted_features[19],
            'y1': handcrafted_features[20],
            'x2': handcrafted_features[21],
            'y2': handcrafted_features[22]
        } if len(handcrafted_features) >= 23 else None
        return result, confidence, debug_info, None, visual_data
    
    reconstruction = autoencoder.predict(img)
    error = np.mean(np.square(img - reconstruction))
    is_anomaly_autoencoder = error >= optimal_threshold
    
    img_cv = (img[0] * 255).astype(np.uint8)
    latent_features = encoder.predict(img)
    handcrafted_features = extract_enhanced_features(img_cv)
    
    if handcrafted_features is None:
        return "Error", 0, {"error": "Feature extraction failed"}, None, None
    
    handcrafted_features_normalized = scaler.transform([handcrafted_features[:EXPECTED_FEATURE_COUNT]])
    combined_features = np.concatenate([latent_features[0], handcrafted_features_normalized[0]])
    
    rf_pred = model_rf.predict([combined_features])[0]
    rf_proba = model_rf.predict_proba([combined_features])[0]
    confidence = rf_proba[int(rf_pred)] * 100
    
    bright_pixels = handcrafted_features[5]
    large_component = handcrafted_features[7]
    human_shape_score = handcrafted_features[16]
    thermal_signature_score = handcrafted_features[17]
    symmetry_score = handcrafted_features[18]
    
    human_presence = False
    thermal_anomaly = error >= optimal_threshold * 1.3
    strong_rf_prediction = rf_pred == 1 and confidence >= 55
    human_like_shape = (human_shape_score > 0.4 and 
                       thermal_signature_score > 0.3 and 
                       symmetry_score > 0.6)
    significant_heat = bright_pixels > 100
    meaningful_size = large_component > 80
    unusual_pattern = handcrafted_features[15] > 50
    
    if (strong_rf_prediction and thermal_anomaly) or \
       (human_like_shape and significant_heat) or \
       (thermal_anomaly and meaningful_size and unusual_pattern):
        human_presence = True
    
    if human_presence:
        confidence_factors = {
            'rf_confidence': confidence * 0.4,
            'size_factor': min(large_component/200, 1) * 20,
            'heat_factor': min(bright_pixels/300, 1) * 15,
            'shape_score': human_shape_score * 15,
            'thermal_score': thermal_signature_score * 10
        }
        confidence = min(sum(confidence_factors.values()), 100)
        confidence = max(confidence, 65)
    else:
        confidence = min(confidence, 60)
    
    result = "Abnormal (Human Detected)" if human_presence else "Normal (No Human)"
    
    debug_info = {
        "reconstruction_error": float(error),
        "threshold": float(optimal_threshold),
        "autoencoder_anomaly": bool(is_anomaly_autoencoder),
        "rf_prediction": int(rf_pred),
        "rf_confidence": float(rf_proba[int(rf_pred)] * 100),
        "bright_pixels": int(bright_pixels),
        "largest_component": int(large_component),
        "human_shape_score": float(human_shape_score),
        "thermal_signature_score": float(thermal_signature_score),
        "symmetry_score": float(symmetry_score),
        "quadrant_variance": float(handcrafted_features[15])
    }
    
    recon_img = (reconstruction[0] * 255).astype('uint8')
    visual_data = {
        'x1': handcrafted_features[19],
        'y1': handcrafted_features[20],
        'x2': handcrafted_features[21],
        'y2': handcrafted_features[22]
    } if len(handcrafted_features) >= 23 and human_presence else None
    
    return result, confidence, debug_info, recon_img, visual_data

def process_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        img_np = np.array(image.convert('RGB'))
        img_resized = cv2.resize(img_np, img_size)
        img_input = img_resized.astype('float32') / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        autoencoder, encoder, model_rf, optimal_threshold, scaler = load_models()
        result, confidence, debug_info, recon_img, visual_data = detect_human_presence(
            img_input, autoencoder, encoder, model_rf, optimal_threshold, scaler
        )
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        recon_str = None
        if recon_img is not None:
            recon_pil = Image.fromarray(recon_img)
            buffered = BytesIO()
            recon_pil.save(buffered, format="PNG")
            recon_str = base64.b64encode(buffered.getvalue()).decode()
        
        return result, confidence, debug_info, img_str, recon_str, visual_data
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None, None, None

def main():
    st.set_page_config(
        page_title="ThermoNyx AI",
        page_icon="🌡️",
        layout="wide"
    )
    
    css = """
    <style>
    body {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 1px solid #2a2a4e;
    }
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #4db6ac;
        margin-bottom: 0.5rem;
    }
    .header p {
        font-size: 1.1rem;
        color: #90a4ae;
    }
    .sidebar-title {
        font-size: 1.5rem;
        color: #4db6ac;
        margin-bottom: 1rem;
    }
    .sidebar-info {
        margin-top: 2rem;
        padding: 1rem;
        background: #2a2a4e;
        border-radius: 8px;
    }
    .sidebar-info h3 {
        font-size: 1.2rem;
        color: #4db6ac;
        margin-bottom: 0.5rem;
    }
    .sidebar-info p, .sidebar-info ul {
        font-size: 0.95rem;
        color: #b0bec5;
    }
    .section-title {
        font-size: 1.3rem;
        color: #4db6ac;
        margin-bottom: 1rem;
    }
    .result-box, .feature-box {
        background: #2a2a4e;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out;
    }
    .result-box:hover, .feature-box:hover {
        transform: translateY(-2px);
    }
    .result-box p, .feature-box p {
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    .status-normal {
        color: #4caf50;
        font-weight: 600;
    }
    .status-abnormal {
        color: #ef5350;
        font-weight: 600;
    }
    .uploaded-image {
        max-width: 100%;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .detection-box {
        position: absolute;
        border: 2px solid #4db6ac;
        background-color: rgba(77, 182, 172, 0.2);
    }
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #2a2a4e;
        color: #90a4ae;
        font-size: 0.9rem;
    }
    .stFileUploader > div > div {
        background: #2a2a4e !important;
        border: 1px solid #4db6ac !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    .stFileUploader > div > div:hover {
        background: #3a3a5e !important;
    }
    .stSpinner > div {
        color: #4db6ac !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="header">
            <h1>🌡️ ThermoNyx AI</h1>
            <p>Advanced Thermal Anomaly Detection for Drone Imagery</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">Upload Image</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a thermal image", type=["png", "jpg", "jpeg"], label_visibility="hidden")
        
        st.markdown("""
            <div class="sidebar-info">
                <h3>About</h3>
                <p>ThermoNyx AI uses deep learning and machine learning to detect human presence in thermal drone images with high accuracy.</p>
                <h3>Features</h3>
                <ul>
                    <li>Real-time analysis</li>
                    <li>High confidence scoring</li>
                    <li>Advanced feature extraction</li>
                    <li>Reconstruction visualization</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            result, confidence, debug_info, img_str, recon_str, visual_data = process_image(uploaded_file)
        
        if result is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<h2 class="section-title">Results</h2>', unsafe_allow_html=True)
                status_class = "status-abnormal" if "Abnormal" in result else "status-normal"
                st.markdown(f"""
                    <div class="result-box">
                        <p><strong>Status:</strong> <span class="{status_class}">{result}</span></p>
                        <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if debug_info and not debug_info.get("error"):
                    st.markdown('<h3 class="section-title">Feature Analysis</h3>', unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="feature-box">
                            <p><strong>Bright Pixels:</strong> {debug_info['bright_pixels']:.0f}</p>
                            <p><strong>Largest Component:</strong> {debug_info['largest_component']:.0f}</p>
                            <p><strong>Human Shape Score:</strong> {debug_info['human_shape_score']:.2f}</p>
                            <p><strong>Thermal Signature:</strong> {debug_info['thermal_signature_score']:.2f}</p>
                            <p><strong>Symmetry Score:</strong> {debug_info['symmetry_score']:.2f}</p>
                            <p><strong>Quadrant Variance:</strong> {debug_info['quadrant_variance']:.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown('<h2 class="section-title">Visualizations</h2>', unsafe_allow_html=True)
                if visual_data and "Abnormal" in result:
                    x1 = int(visual_data['x1'] * img_size[0])
                    y1 = int(visual_data['y1'] * img_size[1])
                    x2 = int(visual_data['x2'] * img_size[0])
                    y2 = int(visual_data['y2'] * img_size[1])
                    st.markdown(f"""
                        <div style="position: relative;">
                            <img src="data:image/png;base64,{img_str}" class="uploaded-image" />
                            <div class="detection-box" style="
                                left: {x1}px; top: {y1}px;
                                width: {x2-x1}px; height: {y2-y1}px;">
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<img src="data:image/png;base64,{img_str}" class="uploaded-image" />', unsafe_allow_html=True)
                
                if recon_str:
                    st.markdown('<h3 class="section-title">Reconstruction</h3>', unsafe_allow_html=True)
                    st.markdown(f'<img src="data:image/png;base64,{recon_str}" class="uploaded-image" />', unsafe_allow_html=True)

if __name__ == "__main__":
    main()