# 📰 Topic Analyzer

An AI-powered News Intelligence Platform built using Python, Machine Learning, NLP, and Streamlit.

The application can analyze news articles from either pasted text or a URL and provide:

* News Category Classification
* Topic Discovery using LDA Topic Modeling
* Article Summarization
* Sentiment Analysis
* Keyword Extraction
* Named Entity Recognition
* Interactive Visualizations

## 🚀 Live Demo

🌐 https://topic-analyser.streamlit.app/

---

## Features

### News Classification

Predicts the category of a news article using a trained Machine Learning model.

Categories include:

* Business
* Politics
* Sports
* Technology
* Entertainment

### Topic Modeling

Uses Latent Dirichlet Allocation (LDA) to discover hidden topics within news articles.

### Article Summarization

Generates a concise summary of lengthy articles to quickly understand the main points.

### Sentiment Analysis

Determines whether the article sentiment is:

* Positive
* Negative
* Neutral

### Keyword Extraction

Identifies the most important keywords from the article using TF-IDF.

### Named Entity Recognition

Extracts important entities such as:

* People
* Organizations
* Locations

### URL-Based Article Analysis

Users can simply paste a news article URL and the application will automatically extract and analyze the content.

---

## Technology Stack

### Frontend

* Streamlit

### Backend

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* TF-IDF Vectorization
* Latent Dirichlet Allocation (LDA)

### Natural Language Processing

* NLTK
* TextBlob

### Web Scraping

* Requests
* BeautifulSoup

### Visualization

* Matplotlib
* Seaborn

---

## Project Structure

topic-analysis/

├── app.py

├── data/

│   ├── bbc-news-data.csv

│   ├── bbc_news_cleaned.csv

│   └── bbc_news_topics.csv

├── models/

│   ├── classifier.pkl

│   ├── tfidf_vectorizer.pkl

│   ├── lda.pkl

│   └── count_vectorizer.pkl

├── src/

│   ├── preprocessing.py

│   ├── eda.py

│   ├── eda_visualization.py

│   ├── tfidf_analysis.py

│   ├── topic_modeling.py

│   ├── news_classifier.py

│   ├── summarizer.py

│   ├── sentiment.py

│   └── entity_extraction.py

├── requirements.txt

└── README.md

---

## Machine Learning Pipeline

### Data Collection

BBC News Dataset

### Data Preprocessing

* Lowercasing
* URL Removal
* Number Removal
* Tokenization
* Stopword Removal
* Lemmatization

### Feature Engineering

TF-IDF Vectorization

### Classification

Machine Learning model trained to predict article categories.

### Topic Modeling

LDA (Latent Dirichlet Allocation) used for discovering hidden topics.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/topic-analysis.git

cd topic-analysis
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Example Workflow

1. Paste a news article or URL.
2. Click Analyze.
3. View:

   * Predicted Category
   * Classification Confidence
   * Topic Distribution
   * Article Summary
   * Sentiment Analysis
   * Keywords
   * Named Entities

---

## Future Enhancements

* Transformer-based Summarization
* BERT Topic Modeling
* Multi-language Support
* Real-time News Feed Analysis
* News Recommendation System
* Interactive Dashboard Analytics

---

## Author

**Swarit Dixit**

Developed as a Machine Learning and Natural Language Processing project demonstrating end-to-end text analytics, topic modeling, classification, and deployment using Streamlit.
