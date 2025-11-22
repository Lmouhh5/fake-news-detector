import pandas as pd
import os
import logging
from pathlib import Path
from typing import Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(file_path: str, expected_columns: Tuple[str, str] = ('text', 'label')) -> pd.DataFrame:
    """
    Load and validate the dataset.

    Parameters:
        file_path (str): Path to the CSV dataset.
        expected_columns (Tuple[str, str]): The expected column names for text and label.

    Returns:
        pd.DataFrame: Cleaned DataFrame with expected columns only.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Expected columns {expected_columns} not found. Found columns: {df.columns.tolist()}")

    # Keep only necessary columns and drop rows with nulls
    df = df[list(expected_columns)].copy()
    initial_shape = df.shape
    df = df.dropna()
    logger.info(f"Dropped {initial_shape[0] - df.shape[0]} rows with missing values")

    # Standardize labels if needed
    if df['label'].dtype == object:
        try:
            df['label'] = pd.to_numeric(df['label'], errors='coerce')
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)
        except Exception as e:
            logger.error("Error converting labels to numeric")
            raise e

    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")

    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str = 'data/processed/cleaned_data.csv') -> None:
    """
    Save cleaned data to the specified path.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Output file path.
    """
    # Use an absolute path for the output file.
    abs_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", output_path))
    Path(abs_output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(abs_output_path, index=False)
    logger.info(f"Cleaned data saved to: {abs_output_path}")


# Optional usage example
if __name__ == "__main__":
    try:
        # Use an absolute path to load the dataset
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "data.csv"))
        data = load_data(file_path)
        save_cleaned_data(data)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
