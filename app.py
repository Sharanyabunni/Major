import os
import numpy as np
import joblib
from PIL import Image
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments_hu
from skimage.color import rgb2gray
from flask import Flask, render_template, request, jsonify, redirect, url_for, make_response
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import json
import datetime

app = Flask(__name__)

# Load the trained model
model = joblib.load('voting_classifier_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
class_names = joblib.load('class_names.pkl')

# Define risk levels and descriptions for educational purposes
RISK_LEVELS = {
    "High Risk": {
        "description": "High probability of coronary artery disease or acute coronary syndrome.",
        "recommendations": [
            "Immediate consultation with a cardiologist is recommended",
            "Further diagnostic tests may be necessary (e.g., coronary angiography)",
            "Lifestyle modifications and medication adherence are crucial"
        ],
        "color": "#e74c3c"
    },
    "Moderate Risk": {
        "description": "Moderate probability of coronary artery disease or acute coronary syndrome.",
        "recommendations": [
            "Consultation with a healthcare provider is recommended",
            "Consider additional non-invasive cardiac testing",
            "Lifestyle modifications to reduce cardiovascular risk factors"
        ],
        "color": "#f39c12"
    },
    "Low Risk": {
        "description": "Low probability of coronary artery disease or acute coronary syndrome.",
        "recommendations": [
            "Continue regular health check-ups",
            "Maintain heart-healthy lifestyle",
            "Monitor for any changes in symptoms"
        ],
        "color": "#2ecc71"
    }
}

# Feature extraction functions
def extract_color_histograms(image):
    """Extract color histograms from an RGB image"""
    hist_r = np.histogram(image[:,:,0], bins=32, range=(0,256))[0]
    hist_g = np.histogram(image[:,:,1], bins=32, range=(0,256))[0]
    hist_b = np.histogram(image[:,:,2], bins=32, range=(0,256))[0]
    
    # Combine histograms
    hist_features = np.concatenate([hist_r, hist_g, hist_b])
    
    # Add some basic statistics (mean and std for each channel)
    mean_r = np.mean(image[:,:,0])
    mean_g = np.mean(image[:,:,1])
    mean_b = np.mean(image[:,:,2])
    std_r = np.std(image[:,:,0])
    std_g = np.std(image[:,:,1])
    std_b = np.std(image[:,:,2])
    
    stats_features = np.array([mean_r, mean_g, mean_b, std_r, std_g, std_b])
    
    return np.concatenate([hist_features, stats_features])

def extract_hu_moments(image):
    """Extract Hu moments from an image"""
    gray = rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)
    hu_moments = moments_hu(gray)
    return hu_moments

def extract_haralick_features(image):
    """Extract Haralick texture features from an image"""
    gray = rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)
    
    # Calculate GLCM with different distances and angles
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # Calculate Haralick properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    
    for prop in properties:
        feature = graycoprops(glcm, prop).flatten()
        features.append(feature)
    
    return np.array(features).flatten()

def extract_color_texture(image):
    """Extract color texture features using LBP or similar methods"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Get individual channels
    h, s, v = cv2.split(hsv)
    
    # Apply LBP-like texture analysis on each channel
    features = []
    for channel in [h, s, v]:
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = np.mean(((channel - mean) / (std + 1e-10)) ** 3)
        kurtosis = np.mean(((channel - mean) / (std + 1e-10)) ** 4) - 3
        features.extend([mean, std, skewness, kurtosis])
    
    return np.array(features)

def extract_all_features(X_data):
    """Extract all features from a dataset of images"""
    all_features = []
    
    for img in X_data:
        # Extract different feature types
        color_hist = extract_color_histograms(img)
        hu_moments = extract_hu_moments(img)
        haralick = extract_haralick_features(img)
        color_tex = extract_color_texture(img)
        
        # Combine all features
        combined = np.concatenate([color_hist, hu_moments, haralick, color_tex])
        all_features.append(combined)
    
    features_array = np.array(all_features)
    return features_array

def create_probability_chart(probabilities, predicted_class):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Sort probabilities for better visualization
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    # Create color palette
    colors = ['#3498db'] * len(sorted_probs)
    predicted_idx = sorted_classes.index(class_names[predicted_class])
    colors[predicted_idx] = '#e74c3c'
    
    # Create bar chart
    bars = plt.bar(range(len(sorted_probs)), sorted_probs, color=colors)
    plt.xticks(range(len(sorted_probs)), sorted_classes, rotation=45, ha='right')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Prediction Probabilities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    chart = base64.b64encode(image_png).decode('utf-8')
    return chart

def create_feature_importance_chart(features, top_n=10):
    # This is a placeholder - in a real application, you would use feature importance from the model
    # For demonstration, we'll create a simulated feature importance chart
    
    # Get feature names (simplified for demonstration)
    feature_categories = [
        "Color Histogram", "Statistical", "Hu Moments", 
        "Haralick Texture", "Color Texture"
    ]
    
    # Create simulated importance scores
    importance = np.random.rand(len(feature_categories))
    importance = importance / np.sum(importance)  # Normalize
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Sort by importance
    sorted_idx = np.argsort(importance)
    
    # Create horizontal bar chart
    plt.barh(
        [feature_categories[i] for i in sorted_idx],
        [importance[i] for i in sorted_idx],
        color='#3498db'
    )
    
    plt.xlabel('Relative Importance', fontsize=12)
    plt.title('Feature Category Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot to a base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    chart = base64.b64encode(image_png).decode('utf-8')
    return chart

def determine_risk_level(confidence):
    """Determine risk level based on prediction confidence"""
    if confidence >= 0.8:
        return "High Risk"
    elif confidence >= 0.5:
        return "Moderate Risk"
    else:
        return "Low Risk"

def generate_report_text(data):
    """Generate a text report from prediction data"""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""CAROTID ULTRASOUND ANALYSIS REPORT
Generated on: {date_str}

PREDICTION RESULTS
-----------------
Predicted Class: {data['prediction']}
Confidence: {data['confidence']*100:.2f}%
Assessment: {data['risk_description']}

RECOMMENDATIONS
--------------
"""
    
    for i, rec in enumerate(data['risk_recommendations'], 1):
        report += f"{i}. {rec}\n"
    
    report += """
DISCLAIMER
---------
This report is generated by an automated system and should be interpreted by a qualified healthcare professional. 
The results are not a definitive diagnosis and should be considered alongside other clinical data and professional judgment.

CardioPredict - Advanced CAD & ACS Prediction System
"""
    
    return report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/technology')
def technology():
    return render_template('technology.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'})
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'})
        
        try:
            # Read image without saving to disk
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # Resize image to match training data
            img_resized = img.resize((64, 64), Image.LANCZOS)
            img_array = np.array(img_resized)
            
            # Extract features
            img_reshaped = np.expand_dims(img_array, axis=0)
            features = extract_all_features(img_reshaped)
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            predicted_class = class_names[prediction]
            
            # Get probabilities
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]
            
            # Determine risk level
            risk_level = determine_risk_level(confidence)
            risk_info = RISK_LEVELS[risk_level]
            
            # Create probability chart
            prob_chart = create_probability_chart(probabilities, prediction)
            
            # Create feature importance chart
            feature_chart = create_feature_importance_chart(features)
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Create a unique session ID for this prediction
            session_id = base64.b64encode(os.urandom(16)).decode('utf-8')
            
            result_data = {
                'success': True,
                'prediction': predicted_class,
                'confidence': float(confidence),
                'risk_level': risk_level,
                'risk_description': risk_info['description'],
                'risk_recommendations': risk_info['recommendations'],
                'risk_color': risk_info['color'],
                'image': img_str,
                'prob_chart': prob_chart,
                'feature_chart': feature_chart,
                'session_id': session_id
            }
            
            return jsonify(result_data)
            
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/download-report', methods=['POST'])
def download_report():
    try:
        data = request.json
        report_text = generate_report_text(data)
        
        # Create a response with the report text
        response = make_response(report_text)
        response.headers["Content-Disposition"] = "attachment; filename=carotid_analysis_report.txt"
        response.headers["Content-Type"] = "text/plain"
        
        return response
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

