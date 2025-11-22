# Fake News Detector Web Application

A web application that uses machine learning to detect fake news articles. The application analyzes articles using a combination of TF-IDF features and linguistic features to make predictions.

## Demo



https://github.com/user-attachments/assets/0f98434f-850e-4dcf-ab44-05a3827efa81



## Features

- Real-time article analysis
- Modern, responsive user interface
- Detailed linguistic feature analysis
- Confidence scores for predictions
- Support for both short and long articles

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data and spaCy model:
```bash
python -c "import nltk; nltk.download('all')"

python -m spacy download en_core_web_sm
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter or paste an article text in the input area and click "Analyze Article" to get the prediction.

## Model Information

The application uses a combined model that incorporates:
- TF-IDF features for text analysis
- Linguistic features including:
  - Average word length
  - Average sentence length
  - Noun/verb/adjective ratios
  - Punctuation ratio
  - Unique word ratio
  - Stopword ratio

## Project Structure

```
.
├── app.py              # Flask application
├── models/            # Directory containing trained models
│   ├── combined_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── feature_scaler.pkl
├── templates/         # HTML templates
│   └── index.html
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
