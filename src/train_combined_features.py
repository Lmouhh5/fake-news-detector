import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from scipy.sparse import load_npz, hstack
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DenseTransformer:
    """Converts sparse matrix to dense (for models like Naive Bayes, KNN)."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def get_models():
    return {
        'LogisticRegression': Pipeline([
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        'GaussianNB': Pipeline([
            ('to_dense', DenseTransformer()),
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ]),
        'LinearSVC': Pipeline([
            ('clf', LinearSVC(max_iter=1000, random_state=42))
        ]),
        'KNN': Pipeline([
            ('to_dense', DenseTransformer()),
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ])
    }


def train_combined_features(tfidf_dir='data/processed/vectorized/tfidf',
                          count_dir='data/processed/vectorized/count',
                          hash_dir='data/processed/vectorized/hash',
                          ling_feat_path='data/processed/linguistic_features_aligned.csv',
                          output_base_dir='models/combined'):
    try:
        # Load aligned linguistic features
        logger.info(f"Loading aligned linguistic features from: {ling_feat_path}")
        ling_df = pd.read_csv(ling_feat_path)
        logger.info(f"Linguistic features shape: {ling_df.shape}")

        # Get labels for stratification
        y = ling_df['label'].values
        indices = np.arange(len(ling_df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
        logger.info(f"Train set size: {len(train_idx)}, Test set size: {len(test_idx)}")

        # Prepare linguistic features
        X_ling = ling_df.drop(columns=['label'])
        X_ling_train = X_ling.iloc[train_idx].values
        X_ling_test = X_ling.iloc[test_idx].values
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Create and fit feature scaler on training data
        logger.info("Creating and fitting feature scaler...")
        feature_scaler = StandardScaler()
        X_ling_train_scaled = feature_scaler.fit_transform(X_ling_train)
        X_ling_test_scaled = feature_scaler.transform(X_ling_test)
        
        # Save the feature scaler
        scaler_path = Path(output_base_dir) / 'feature_scaler.joblib'
        joblib.dump(feature_scaler, scaler_path)
        logger.info(f"Saved feature scaler to: {scaler_path}")

        # Dictionary to store feature combinations
        feature_dirs = {
            'tfidf': Path(tfidf_dir),
            'count': Path(count_dir),
            'hash': Path(hash_dir)
        }

        # Train models for each feature combination
        for feat_type, feat_dir in feature_dirs.items():
            logger.info(f"\nTraining models with {feat_type.upper()} + Linguistic features")
            output_dir = Path(output_base_dir) / feat_type
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir.absolute()}")

            # Load and align feature matrix
            logger.info(f"Loading {feat_type} matrix...")
            X_full = load_npz(feat_dir / 'X_full.npz')
            logger.info(f"Full {feat_type} matrix shape: {X_full.shape}")
            X_full = X_full[:len(ling_df)]
            logger.info(f"Aligned {feat_type} matrix shape: {X_full.shape}")

            # Split features
            X_train = X_full[train_idx]
            X_test = X_full[test_idx]
            logger.info(f"X_{feat_type}_train shape: {X_train.shape}")
            logger.info(f"X_{feat_type}_test shape: {X_test.shape}")

            # Combine features (using scaled linguistic features)
            logger.info("Combining features...")
            X_combined_train = hstack([X_train, X_ling_train_scaled])
            X_combined_test = hstack([X_test, X_ling_test_scaled])
            logger.info(f"Combined X_train shape: {X_combined_train.shape}")
            logger.info(f"Combined X_test shape: {X_combined_test.shape}")

            # Train models
            models = get_models()
            results = []
            for name, model in models.items():
                try:
                    logger.info(f"\nTraining {feat_type} + Linguistic model: {name}")
                    model.fit(X_combined_train, y_train)
                    y_pred = model.predict(X_combined_test)

                    acc = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred)

                    logger.info(f"Accuracy: {acc:.4f}")
                    logger.info(report)

                    # Save model
                    model_path = output_dir / f"{name}.joblib"
                    logger.info(f"Saving model to: {model_path}")
                    joblib.dump(model, model_path)

                    # Save report
                    report_path = output_dir / f"{name}_report.txt"
                    logger.info(f"Saving report to: {report_path}")
                    with open(report_path, 'w') as f:
                        f.write(f"Accuracy: {acc:.4f}\n\n{report}")

                    results.append({'Model': name, 'Accuracy': acc})
                except Exception as e:
                    logger.error(f"Error training {name}: {str(e)}", exc_info=True)
                    continue

            # Save results summary for this feature combination
            if results:
                df = pd.DataFrame(results)
                summary_path = output_dir / f'{feat_type}_model_comparison.csv'
                logger.info(f"Saving results summary to: {summary_path}")
                df.to_csv(summary_path, index=False)
                logger.info(f"Saved {feat_type} model comparison.")
            else:
                logger.error(f"No models were successfully trained for {feat_type}!")

    except Exception as e:
        logger.error(f"Error in train_combined_features: {str(e)}", exc_info=True)
        raise


def main():
    train_combined_features()


if __name__ == '__main__':
    main() 