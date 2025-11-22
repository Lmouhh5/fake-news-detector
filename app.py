# =============================================================================
# Fake News Detection System
# This application uses multiple machine learning models to analyze news articles
# and determine whether they are likely to be real or fake news.
# =============================================================================

from flask import Flask, render_template, request, jsonify
import logging
from pathlib import Path
import joblib
import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

# Add src directory to Python path for custom module imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import prediction functionality from predict.py module
from predict import (
    load_models_and_tools,
    prepare_input,
    get_prediction_confidence,
    BEST_MODELS
)

# Configure logging to track application behavior and debug issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

class DenseTransformer:
    """
    Utility class to convert sparse matrices to dense format.
    Required for models like Naive Bayes and KNN that don't work with sparse matrices.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, 'toarray'):
            return X.toarray()
        return X  # If already dense, return as is

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# Global variables to store loaded models and preprocessing tools
models = {}          # Dictionary to store all trained models
vectorizers = {}     # Dictionary to store text vectorizers for each model
scaler = None        # Scaler for feature normalization

def initialize_models():
    """
    Initialize and load all machine learning models and preprocessing tools.
    This function runs at application startup to ensure all required components
    are available before accepting requests.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global models, vectorizers, scaler
    try:
        logger.info("Loading models and tools...")
        models, vectorizers, scaler = load_models_and_tools()
        if not models:
            logger.error("No models could be loaded!")
            return False
        logger.info(f"Successfully loaded {len(models)} models")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

@app.route('/')
def home():
    """Render the main application interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests for news article analysis.
    
    This endpoint:
    1. Receives news article text from the client
    2. Processes the text through multiple ML models
    3. Combines predictions using a weighted ensemble approach
    4. Returns detailed prediction results and confidence scores
    
    Returns:
        JSON response containing:
        - Individual model predictions and confidences
        - Ensemble prediction and confidence
        - Model agreement statistics
    """
    try:
        # Extract and validate input text
        data = request.get_json()
        logger.info(f"Received request data: {data}")
        
        if not data or 'text' not in data:
            logger.error("Missing text in request data")
            return jsonify({
                'error': 'Please provide text to analyze.'
            }), 400

        text = data['text'].strip()
        if not text:
            logger.error("Empty text in request")
            return jsonify({
                'error': 'Please provide some text to analyze.'
            }), 400

        logger.info(f"Processing text: {text[:100]}...")

        # Process text through each model in the ensemble
        results = []
        for model_name, model in models.items():
            try:
                # Get model configuration
                model_info = BEST_MODELS[model_name]
                logger.info(f"Processing model: {model_name}")
                
                # Check if model requires dense feature representation
                needs_dense = model_name in ['GaussianNB (TF-IDF)', 'KNN (Count)']
                
                # Preprocess input text for current model
                features = prepare_input(
                    text, 
                    vectorizers[model_name], 
                    scaler,
                    needs_dense=needs_dense
                )
                
                if features is None:
                    logger.warning(f"Could not prepare features for {model_name}")
                    continue
                
                # Get model prediction and confidence
                prediction = model.predict(features)[0]
                confidence = get_prediction_confidence(model, features)
                
                # Store individual model results
                result = {
                    'model': model_name,
                    'description': model_info['description'],
                    'accuracy': model_info['accuracy'],
                    'weight': model_info['weight'],
                    'prediction': 'REAL' if prediction == 1 else 'FAKE',
                    'confidence': float(confidence)
                }
                logger.info(f"Model {model_name} prediction: {result}")
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error with {model_name}: {str(e)}", exc_info=True)
                continue

        if not results:
            logger.error("No predictions could be made")
            return jsonify({
                'error': 'Could not make predictions. Please try again.'
            }), 500

        # Calculate weighted ensemble prediction
        # Combine individual model predictions using their weights and confidence scores
        real_scores = [
            result['weight'] * result['confidence'] 
            for result in results 
            if result['prediction'] == 'REAL'
        ]
        fake_scores = [
            result['weight'] * result['confidence'] 
            for result in results 
            if result['prediction'] == 'FAKE'
        ]
        total_weight = sum(result['weight'] for result in results)
        
        # Normalize scores to get final prediction probabilities
        real_score = float(sum(real_scores) / total_weight if real_scores else 0)
        fake_score = float(sum(fake_scores) / total_weight if fake_scores else 0)
        
        # Determine final ensemble prediction
        ensemble_prediction = 'REAL' if real_score > fake_score else 'FAKE'
        ensemble_confidence = float(max(real_score, fake_score))

        # Prepare detailed response with all results
        response = {
            'predictions': results,
            'ensemble': {
                'prediction': ensemble_prediction,
                'confidence': ensemble_confidence,
                'real_score': real_score,
                'fake_score': fake_score,
                'model_agreement': {
                    'real': len(real_scores),
                    'fake': len(fake_scores)
                }
            }
        }
        
        logger.info(f"Sending response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred during prediction. Please try again.'
        }), 500

if __name__ == '__main__':
    # Initialize models before starting the server
    if initialize_models():
        app.run(debug=True)
    else:
        logger.error("Failed to initialize models. Application will not start.") 