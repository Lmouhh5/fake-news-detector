import pandas as pd
import logging
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    HashingVectorizer
)
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def vectorize_texts(X_train_text, X_test_text, method='tfidf', max_features=5000, vectorizer=None):
    """
    Vectorize text using specified method.
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
    elif method == 'count':
        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
    elif method == 'hash':
        vectorizer = HashingVectorizer(
            n_features=max_features,
            ngram_range=(1, 2),
            alternate_sign=False,
            stop_words='english'
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    logger.info(f"Vectorizing with {method}...")
    X_train_vec = vectorizer.fit_transform(X_train_text) if method != 'hash' else vectorizer.transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    return vectorizer, X_train_vec, X_test_vec


def save_vectorized_data(method_name, vectorizer, X_train, X_test, y_train, y_test, X_full=None, base_dir='data/processed/vectorized'):
    output_dir = Path(base_dir) / method_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if X_full is not None:
        save_npz(output_dir / 'X_full.npz', X_full)
        logger.info(f"Saved full {method_name} matrix")

    save_npz(output_dir / 'X_train.npz', X_train)
    save_npz(output_dir / 'X_test.npz', X_test)
    joblib.dump(y_train, output_dir / 'y_train.joblib')
    joblib.dump(y_test, output_dir / 'y_test.joblib')

    if method_name != 'hash':
        joblib.dump(vectorizer, output_dir / f"{method_name}_vectorizer.joblib")

    logger.info(f"{method_name.upper()} vectorized data saved to: {output_dir}")


def main(input_path='data/processed/preprocessed_data.csv', ling_path='data/processed/linguistic_features.csv'):
    try:
        logger.info(f"Loading input data from: {input_path}")
        df = pd.read_csv(input_path)
        ling_df = pd.read_csv(ling_path)

        if 'clean_text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Missing 'clean_text' or 'label' column.")
        if 'label' not in ling_df.columns:
            raise ValueError("Missing 'label' column in linguistic features.")

        # Drop rows with NaN values in clean_text and label in both dataframes
        mask_text = df['clean_text'].notna() & df['label'].notna()
        mask_ling = ling_df['label'].notna() & ling_df.drop(columns=['label']).notna().all(axis=1)
        mask = mask_text & mask_ling
        df = df[mask].reset_index(drop=True)
        ling_df = ling_df[mask].reset_index(drop=True)
        logger.info(f"Final aligned data shape: {df.shape}, linguistic features shape: {ling_df.shape}")

        # Save the aligned linguistic features for use in training
        ling_df.to_csv('data/processed/linguistic_features_aligned.csv', index=False)
        logger.info("Saved aligned linguistic features to data/processed/linguistic_features_aligned.csv")

        X = df['clean_text']
        y = df['label']

        for method in ['tfidf', 'count', 'hash']:
            # First vectorize the full dataset
            vectorizer, X_full, _ = vectorize_texts(X, X, method)
            # Then split and vectorize train/test
            X_train_text, X_test_text, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )
            _, X_train_vec, X_test_vec = vectorize_texts(X_train_text, X_test_text, method, vectorizer=vectorizer)
            save_vectorized_data(method, vectorizer, X_train_vec, X_test_vec, y_train, y_test, X_full)

        logger.info("All vectorizations complete.")

    except Exception as e:
        logger.error(f"Error during vectorization: {e}")
        raise


if __name__ == '__main__':
    main() 