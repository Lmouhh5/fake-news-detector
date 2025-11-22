import pandas as pd
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Remove stopwords and apply lemmatization. Keeps original casing and punctuation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.lower() not in STOPWORDS]
    lemmatized = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Preprocess a DataFrame by removing nulls and applying text preprocessing.
    """
    logger.info(f"Starting preprocessing on {len(df)} rows...")

    # Drop rows with missing values in text
    df = df.dropna(subset=[text_column])
    logger.info(f"Remaining rows after dropping nulls: {len(df)}")

    # Preprocess text
    df = df.copy()
    df['clean_text'] = df[text_column].apply(preprocess_text)

    logger.info("Preprocessing complete.")
    return df


def main(input_path: str = 'data/processed/cleaned_data.csv',
         output_path: str = 'data/processed/preprocessed_data.csv'):
    try:
        # Use absolute paths for input and output files.
        abs_input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", input_path))
        abs_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", output_path))
        logger.info(f"Loading data from: {abs_input_path}")
        df = pd.read_csv(abs_input_path)
        df_processed = preprocess_dataframe(df, text_column='text')
        Path(abs_output_path).parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(abs_output_path, index=False)
        logger.info(f"Preprocessed data saved to: {abs_output_path}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


if __name__ == '__main__':
    main()
