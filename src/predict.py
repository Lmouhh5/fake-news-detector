import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from text_preprocessor import preprocess_text
from linguistic_feature_extractor import extract_25_paper_features
from scipy.sparse import csr_matrix
import warnings

# Filter out scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DenseTransformer:
    """Converts sparse matrix to dense (for models like Naive Bayes, KNN)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            return X
        return X.toarray()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

# Define the best performing models and their paths
BEST_MODELS = {
    'RandomForest (TF-IDF)': {
        'path': 'models/combined/tfidf/RandomForest.joblib',
        'vectorizer': 'data/processed/vectorized/tfidf/tfidf_vectorizer.joblib',
        'accuracy': 0.9164,
        'description': 'Random Forest with TF-IDF features (Best overall accuracy)',
        'needs_dense': False,
        'weight': 1.0  # Higher weight for more accurate models
    },
    'GaussianNB (TF-IDF)': {
        'path': 'models/combined/tfidf/GaussianNB.joblib',
        'vectorizer': 'data/processed/vectorized/tfidf/tfidf_vectorizer.joblib',
        'accuracy': 0.8681,
        'description': 'Gaussian Naive Bayes with TF-IDF features',
        'needs_dense': True,
        'weight': 0.8
    },
    'LinearSVC (Count)': {
        'path': 'models/combined/count/LinearSVC.joblib',
        'vectorizer': 'data/processed/vectorized/count/count_vectorizer.joblib',
        'accuracy': 0.9012,
        'description': 'Linear SVC with Count features',
        'needs_dense': False,
        'weight': 0.9
    },
    'KNN (Count)': {
        'path': 'models/combined/count/KNN.joblib',
        'vectorizer': 'data/processed/vectorized/count/count_vectorizer.joblib',
        'accuracy': 0.6964,
        'description': 'K-Nearest Neighbors with Count features',
        'needs_dense': True,
        'weight': 0.6
    }
}

def load_models_and_tools():
    """Load all models and their corresponding vectorizers."""
    models = {}
    vectorizers = {}
    
    for model_name, model_info in BEST_MODELS.items():
        try:
            logger.info(f"Loading {model_name}...")
            model = joblib.load(model_info['path'])
            vectorizer = joblib.load(model_info['vectorizer'])
            
            # Store model and vectorizer
            models[model_name] = model
            vectorizers[model_name] = vectorizer
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            continue
    
    try:
        logger.info("Loading feature scaler...")
        scaler = joblib.load('models/combined/feature_scaler.joblib')
        logger.info("Successfully loaded feature scaler")
    except Exception as e:
        logger.error(f"Error loading feature scaler: {str(e)}")
        scaler = None
    
    return models, vectorizers, scaler

def prepare_input(text, vectorizer, scaler, needs_dense=False):
    """Prepare input text for prediction using the specified vectorizer and scaler."""
    try:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Extract linguistic features
        ling_features = extract_25_paper_features(processed_text)
        if not ling_features:
            logger.error("Failed to extract linguistic features")
            return None
        
        # Convert linguistic features to numpy array directly
        ling_features_array = np.array([list(ling_features.values())])
        
        # Scale linguistic features if scaler is available
        if scaler is not None:
            ling_features_scaled = scaler.transform(ling_features_array)
        else:
            ling_features_scaled = ling_features_array
        
        # Transform text using vectorizer
        text_features = vectorizer.transform([processed_text])
        
        # Convert to dense if needed
        if needs_dense:
            text_features = text_features.toarray()
            combined_features = np.hstack([text_features, ling_features_scaled])
        else:
            # Keep text features sparse
            combined_features = csr_matrix(np.hstack([text_features.toarray(), ling_features_scaled]))
        
        return combined_features
    except Exception as e:
        logger.error(f"Error preparing input: {str(e)}")
        return None

def get_prediction_confidence(model, features):
    """Get prediction confidence for the model."""
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = max(proba)
        else:
            # For models without predict_proba, use decision function
            decision = model.decision_function(features)[0]
            # Normalize decision function to [0,1] range
            confidence = 1 / (1 + np.exp(-decision))  # Sigmoid
            confidence = max(confidence, 1 - confidence)
            # Adjust confidence based on distance from decision boundary
            confidence = 0.5 + abs(confidence - 0.5)  # Scale to [0.5, 1.0]
        return confidence
    except Exception as e:
        logger.error(f"Error getting prediction confidence: {str(e)}")
        return 0.5  # Return neutral confidence on error

def predict(text, models, vectorizers, scaler):
    """Make predictions using all available models."""
    results = []
    
    for model_name, model in models.items():
        try:
            # Get model info
            model_info = BEST_MODELS[model_name]
            
            # Prepare input for this model
            features = prepare_input(
                text, 
                vectorizers[model_name], 
                scaler,
                needs_dense=model_info['needs_dense']
            )
            if features is None:
                continue
            
            # Get prediction
            prediction = model.predict(features)[0]
            confidence = get_prediction_confidence(model, features)
            
            # Adjust confidence based on model accuracy
            adjusted_confidence = confidence * model_info['accuracy']
            
            results.append({
                'model': model_name,
                'description': model_info['description'],
                'accuracy': model_info['accuracy'],
                'weight': model_info['weight'],
                'prediction': 'REAL' if prediction == 1 else 'FAKE',
                'confidence': adjusted_confidence
            })
            
        except Exception as e:
            logger.error(f"Error with {model_name}: {str(e)}")
            continue
    
    return results

def main():
    """Main function to run the interactive prediction system."""
    print("\n🤖 Fake News Detection System")
    print("="*80)
    print("Loading models and tools...")
    
    # Load all models and tools
    models, vectorizers, scaler = load_models_and_tools()
    
    if not models:
        print("❌ Error: No models could be loaded. Please check the model files.")
        return
    
    print(f"\n✅ Successfully loaded {len(models)} models!")
    print("\nAvailable Models:")
    for model_name, model_info in BEST_MODELS.items():
        if model_name in models:
            print(f"📊 {model_name}: {model_info['description']} (Accuracy: {model_info['accuracy']:.2%})")
    
    while True:
        print("\n" + "="*80)
        print("\nEnter the news text to analyze (or 'quit' to exit):")
        print("(Press Enter twice to submit)")
        
        # Get multiline input
        lines = []
        while True:
            line = input()
            if line == 'quit':
                return
            if line == '':
                break
            lines.append(line)
        
        text = '\n'.join(lines)
        if not text.strip():
            print("❌ Please enter some text to analyze.")
            continue
        
        print("\n🔍 Analyzing text...")
        results = predict(text, models, vectorizers, scaler)
        
        if not results:
            print("❌ Error: Could not make predictions. Please try again.")
            continue
        
        print("\n📊 Prediction Results:")
        print("="*80)
        for result in results:
            print(f"\nModel: {result['model']}")
            print(f"Description: {result['description']}")
            print(f"Training Accuracy: {result['accuracy']:.2%}")
            print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
            print("-"*40)
        
        # Calculate weighted ensemble prediction
        real_score = sum(
            result['weight'] * result['confidence'] 
            for result in results 
            if result['prediction'] == 'REAL'
        )
        fake_score = sum(
            result['weight'] * result['confidence'] 
            for result in results 
            if result['prediction'] == 'FAKE'
        )
        total_weight = sum(result['weight'] for result in results)
        
        # Normalize scores
        real_score = real_score / total_weight
        fake_score = fake_score / total_weight
        
        ensemble_prediction = 'REAL' if real_score > fake_score else 'FAKE'
        ensemble_confidence = max(real_score, fake_score)
        
        print("\n🎯 Ensemble Prediction:")
        print(f"Final Decision: {ensemble_prediction}")
        print(f"Confidence: {ensemble_confidence:.2%}")
        print(f"Weighted Scores: REAL {real_score:.2%} vs FAKE {fake_score:.2%}")

if __name__ == "__main__":
    main() 