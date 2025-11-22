import pandas as pd
import logging
from pathlib import Path
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag
from textblob import TextBlob
import textstat
import os

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NLTK setup
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def count_pos(tags, prefix):
    return len([tag for word, tag in tags if tag.startswith(prefix)])


def extract_25_paper_features(text: str) -> dict:
    """
    Extract 25 linguistic features based on Semwal et al. (2022).
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'f01_num_special_chars': 0,
            'f02_num_uppercase': 0,
            'f03_num_lowercase': 0,
            'f04_num_short_sentences': 0,
            'f05_num_long_sentences': 0,
            'f06_article_count': 0,
            'f07_determiner_count': 0,
            'f08_noun_count': 0,
            'f09_verb_count': 0,
            'f10_adverb_count': 0,
            'f11_num_syllables': 0,
            'f12_word_count': 0,
            'f13_sentence_count': 0,
            'f14_rate_of_noun': 0,
            'f15_negation_count': 0,
            'f16_adjective_count': 0,
            'f17_rate_of_adverb': 0,
            'f18_gunning_fog': 0,
            'f19_coleman_liau': 0,
            'f20_linsear_write': 0,
            'f21_dale_chall': 0,
            'f22_flesch': 0,
            'f23_spache': 0,
            'f24_ari': 0,
            'f25_polarity': 0
        }

    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)

    num_sentences = len(sentences)
    num_words = len(tokens)
    avg_sentence_len = num_words / num_sentences if num_sentences else 0

    short_sent = sum(1 for s in sentences if len(s.split()) <= 5)
    long_sent = sum(1 for s in sentences if len(s.split()) >= 20)

    text_blob = TextBlob(text)
    polarity = text_blob.sentiment.polarity
    subjectivity = text_blob.sentiment.subjectivity

    return {
        # Complexity Features
        'f01_num_special_chars': sum(1 for c in text if not c.isalnum() and not c.isspace()),
        'f02_num_uppercase': sum(1 for c in text if c.isupper()),
        'f03_num_lowercase': sum(1 for c in text if c.islower()),
        'f04_num_short_sentences': short_sent,
        'f05_num_long_sentences': long_sent,

        # Stylometric Features
        'f06_article_count': sum(1 for w in tokens if w.lower() in {'a', 'an', 'the'}),
        'f07_determiner_count': count_pos(tags, 'DT'),
        'f08_noun_count': count_pos(tags, 'NN'),
        'f09_verb_count': count_pos(tags, 'VB'),
        'f10_adverb_count': count_pos(tags, 'RB'),
        'f11_num_syllables': sum(textstat.syllable_count(w) for w in tokens if w.isalpha()),
        'f12_word_count': num_words,
        'f13_sentence_count': num_sentences,
        'f14_rate_of_noun': count_pos(tags, 'NN') / num_words if num_words else 0,
        'f15_negation_count': sum(1 for w in tokens if w.lower() in {'not', 'no', 'never', "n't"}),
        'f16_adjective_count': count_pos(tags, 'JJ'),
        'f17_rate_of_adverb': count_pos(tags, 'RB') / num_words if num_words else 0,

        # Readability
        'f18_gunning_fog': textstat.gunning_fog(text),
        'f19_coleman_liau': textstat.coleman_liau_index(text),
        'f20_linsear_write': textstat.linsear_write_formula(text),
        'f21_dale_chall': textstat.dale_chall_readability_score(text),
        'f22_flesch': textstat.flesch_reading_ease(text),
        'f23_spache': textstat.spache_readability(text),
        'f24_ari': textstat.automated_readability_index(text),

        # Psycho-linguistic
        'f25_polarity': polarity,
        # Subjectivity not included since only polarity is mentioned in table
    }


def extract_features_df(df: pd.DataFrame, text_column: str = 'clean_text') -> pd.DataFrame:
    logger.info("Extracting 25 linguistic features from dataset...")
    features = [extract_25_paper_features(text) for text in df[text_column]]
    return pd.DataFrame(features)


def main(input_path: str = 'data/processed/preprocessed_data.csv',
         output_path: str = 'data/processed/linguistic_features.csv'):
    try:
        # Use absolute paths for input and output files.
        abs_input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", input_path))
        abs_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", output_path))
        df = pd.read_csv(abs_input_path)
        if 'clean_text' not in df.columns:
            raise ValueError("Column 'clean_text' not found in preprocessed data.")
        features = extract_features_df(df)
        final_df = pd.concat([features, df['label']], axis=1)
        Path(abs_output_path).parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(abs_output_path, index=False)
        logger.info(f"Linguistic features saved to: {abs_output_path}")
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise


if __name__ == '__main__':
    main()
