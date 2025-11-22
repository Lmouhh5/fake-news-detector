import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import load_npz, hstack
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
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


def evaluate_model(model, X_test, y_test, model_name, feature_type, output_dir):
    """Evaluate a single model and save its metrics and visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics
    logger.info(f"=== {feature_type.upper()} + Linguistic: {model_name} ===")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Save report
    report_path = Path(output_dir) / f"{feature_type}_{model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"=== {feature_type.upper()} + Linguistic: {model_name} ===\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {feature_type.upper()} + Linguistic - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"{feature_type}_{model_name}_confusion_matrix.png")
    plt.close()

    # ROC Curve (if supported)
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {feature_type.upper()} + Linguistic - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f"{feature_type}_{model_name}_roc_curve.png")
        plt.close()

    return {
        'feature_type': feature_type,
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }


def plot_model_comparison(results_df, output_dir):
    """Create comparison plots for all models and feature types."""
    # Accuracy comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='model', y='accuracy', hue='feature_type')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'accuracy_comparison.png')
    plt.close()

    # F1 Score comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='model', y='f1_score', hue='feature_type')
    plt.title('Model F1 Score Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'f1_score_comparison.png')
    plt.close()


def main(
    model_base_dir='models/combined',
    ling_feat_path='data/processed/linguistic_features_aligned.csv',
    tfidf_dir='data/processed/vectorized/tfidf',
    count_dir='data/processed/vectorized/count',
    hash_dir='data/processed/vectorized/hash',
    output_dir='results/combined'
):
    try:
        # Load linguistic features
        logger.info("Loading linguistic features...")
        ling_df = pd.read_csv(ling_feat_path)
        if 'label' not in ling_df.columns:
            raise ValueError("Linguistic features must contain a 'label' column.")

        # Prepare linguistic features
        X_ling = ling_df.drop(columns=['label'])
        y = ling_df['label']

        # Split data (same as training)
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(ling_df))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
        X_ling_test = X_ling.iloc[test_idx].values
        y_test = y.iloc[test_idx].values

        # Dictionary of feature directories
        feature_dirs = {
            'tfidf': Path(tfidf_dir),
            'count': Path(count_dir),
            'hash': Path(hash_dir)
        }

        all_results = []
        for feat_type, feat_dir in feature_dirs.items():
            logger.info(f"\nEvaluating {feat_type.upper()} + Linguistic models...")
            
            # Load feature matrix
            X_full = load_npz(feat_dir / 'X_full.npz')
            X_full = X_full[:len(ling_df)]  # Align with linguistic features
            X_test = X_full[test_idx]
            
            # Combine features
            X_combined_test = hstack([X_test, X_ling_test])
            
            # Load and evaluate models
            model_dir = Path(model_base_dir) / feat_type
            model_files = list(model_dir.glob("*.joblib"))
            if not model_files:
                logger.warning(f"No models found in {model_dir}")
                continue

            for model_file in model_files:
                model_name = model_file.stem
                logger.info(f"\nEvaluating {feat_type.upper()} + Linguistic model: {model_name}")
                model = joblib.load(model_file)
                result = evaluate_model(model, X_combined_test, y_test, model_name, feat_type, output_dir)
                all_results.append(result)

        # Create and save comparison plots
        if all_results:
            results_df = pd.DataFrame(all_results)
            plot_model_comparison(results_df, output_dir)
            
            # Save comprehensive results
            results_df.to_csv(Path(output_dir) / 'evaluation_summary.csv', index=False)
            logger.info(f"\nSaved evaluation summary to {output_dir}/evaluation_summary.csv")
            
            # Print best model for each feature type
            logger.info("\nBest models by feature type:")
            for feat_type in feature_dirs.keys():
                feat_results = results_df[results_df['feature_type'] == feat_type]
                best_model = feat_results.loc[feat_results['accuracy'].idxmax()]
                logger.info(f"\n{feat_type.upper()} + Linguistic:")
                logger.info(f"Best model: {best_model['model']}")
                logger.info(f"Accuracy: {best_model['accuracy']:.4f}")
                logger.info(f"F1 Score: {best_model['f1_score']:.4f}")
        else:
            logger.error("No models were successfully evaluated!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
