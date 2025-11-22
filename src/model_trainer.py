import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, accuracy_score

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_models():
    """
    Initialize ML models in pipelines.

    Returns:
        dict: Dictionary of model name and scikit-learn pipeline
    """
    return {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'GaussianNB': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ]),
        'LinearSVC': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LinearSVC(max_iter=1000, random_state=42))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ])
    }


def train_and_save_models(X, y, output_dir='models/linguistic'):
    """
    Train and save models on the given dataset.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        output_dir (str): Directory to save trained models
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    models = get_models()

    # Train-test split
    logger.info("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    for name, model in models.items():
        logger.info(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"\nClassification Report:\n{report}")

        results[name] = acc

        # Save model
        model_path = Path(output_dir) / f"{name}.joblib"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to: {model_path}")

    return results


def main(input_path: str = 'data/processed/linguistic_features.csv'):
    try:
        logger.info(f"Loading features from: {input_path}")
        df = pd.read_csv(input_path)

        if 'label' not in df.columns:
            raise ValueError("Data must contain a 'label' column.")

        X = df.drop(columns=['label'])
        y = df['label']

        train_and_save_models(X, y)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
