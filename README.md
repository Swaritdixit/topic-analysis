# 📰 Topic Analyzer

An NLP-powered **News Intelligence Platform** built with Python, Machine Learning, NLP, and Streamlit.

The application analyzes news articles from either **pasted text or a URL** and provides:

- News category classification
- Classification confidence
- Topic modeling
- Extractive article summarization
- Sentiment analysis
- TF-IDF keyword extraction
- Named entity recognition
- Interactive visualizations

🔗 **Live Demo:** https://topic-analyser.streamlit.app/

---

## ✨ Features

### 📰 News Classification

Predicts the category of a news article using a trained **Multinomial Naive Bayes** classifier with TF-IDF features.

The model supports five categories:

- Business
- Politics
- Sports
- Technology
- Entertainment

The application also displays the probability associated with each predicted category.

---

### 🔍 Topic Modeling

Uses **Latent Dirichlet Allocation (LDA)** to discover hidden topics in news articles.

The trained LDA model:

- Uses 5 topics
- Produces a topic distribution for each article
- Displays topic probabilities in the application
- Allows comparison between discovered topics and original news categories

---

### ✂️ Article Summarization

Generates an **extractive summary** from an article.

The summarizer:

1. Splits the article into sentences.
2. Calculates word frequencies.
3. Scores sentences according to the frequency of their words.
4. Selects the highest-scoring sentences.
5. Combines them into a concise summary.

This provides a lightweight summarization approach without requiring a large transformer model.

---

### 😊 Sentiment Analysis

Uses **TextBlob** to calculate the polarity of an article.

The resulting sentiment is classified as:

- Positive
- Negative
- Neutral

The classification is based on polarity thresholds.

---

### 🔑 Keyword Extraction

Uses TF-IDF scores to identify important words in an article.

The application extracts the highest-scoring terms and displays them as the article's top keywords.

---

### 🧑‍💼 Named Entity Recognition

Uses NLTK's named entity recognition pipeline to identify entities such as:

- People
- Organizations
- Locations
- Other recognized named entities

The detected entities are displayed in a structured table.

---

### 🌐 URL-Based Article Analysis

Users can provide either:

- Direct article text
- A news article URL

For URL input, the application:

```text
Article URL
     │
     ▼
HTTP Request
     │
     ▼
BeautifulSoup
     │
     ▼
Extract <p> Elements
     │
     ▼
Article Text
     │
     ▼
NLP Pipeline
```

The extracted article is then passed through the same analysis pipeline.

---

## 📊 Dataset

The project uses the **BBC News Dataset**.

The dataset contains:

**2,225 news articles**

| Category | Articles |
|---|---:|
| Sport | 511 |
| Business | 510 |
| Politics | 417 |
| Technology | 401 |
| Entertainment | 386 |
| **Total** | **2,225** |

Each article contains information including:

- Category
- Filename
- Title
- Content

---

## 🧠 Machine Learning Pipeline

The overall NLP pipeline is:

```text
BBC News Dataset
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Text Preprocessing
       │
       ├── Lowercasing
       ├── URL Removal
       ├── Number Removal
       ├── Tokenization
       ├── Stopword Removal
       └── Lemmatization
       │
       ▼
Cleaned Text
       │
       ├───────────────────┐
       │                   │
       ▼                   ▼
   TF-IDF             Count Vectorizer
       │                   │
       ▼                   ▼
Multinomial NB            LDA
       │                   │
       ▼                   ▼
Classification        Topic Modeling
```

Additional NLP analysis is performed on the original article text:

```text
Original Article
       │
       ├── Extractive Summarization
       ├── Sentiment Analysis
       └── Named Entity Recognition
```

---

## 🔬 Text Preprocessing

The preprocessing pipeline performs:

### 1. Lowercasing

Converts all text to lowercase.

```text
"Technology News"
        ↓
"technology news"
```

### 2. URL Removal

URLs are removed using regular expressions.

### 3. Number Removal

Numeric characters are removed.

### 4. Tokenization

The article is split into individual words using NLTK.

### 5. Stopword Removal

Common English stopwords are removed.

Examples:

```text
the
is
and
of
to
```

### 6. Lemmatization

Words are reduced to their base form using NLTK's `WordNetLemmatizer`.

---

## 📈 TF-IDF Classification

The classification pipeline uses **TF-IDF vectorization** with a maximum vocabulary size of 5,000 features.

```text
Cleaned Article
      │
      ▼
TF-IDF Vectorizer
      │
      ▼
Feature Vector
      │
      ▼
Multinomial Naive Bayes
      │
      ▼
Predicted Category
```

The trained model and vectorizer are saved using `joblib`.

Saved artifacts:

```text
models/
├── classifier.pkl
└── tfidf_vectorizer.pkl
```

---

## 🧩 LDA Topic Modeling

Topic discovery uses **Latent Dirichlet Allocation**.

The project uses:

```text
Number of Topics = 5
```

The LDA pipeline is:

```text
Cleaned Articles
      │
      ▼
Count Vectorization
      │
      ▼
Document-Term Matrix
      │
      ▼
LDA
      │
      ▼
5 Topic Distributions
```

The trained artifacts are stored as:

```text
models/
├── lda.pkl
└── count_vectorizer.pkl
```

The project also generates:

- Topic distribution
- Category vs. topic comparison
- Topic percentage distribution
- Top words for each topic
- Category-topic heatmap

---

## 📊 Exploratory Data Analysis

The project includes exploratory analysis of the BBC News dataset.

The EDA pipeline examines:

- Dataset dimensions
- Column information
- Missing values
- Category distribution
- Article length
- Average article length by category
- Word frequency

Visualizations include:

- Category distribution
- Article length distribution
- Average article length by category
- Word cloud

---

## 🏗️ Project Structure

```text
topic-analysis/
│
├── app.py
│
├── data/
│   ├── bbc-news-data.csv
│   ├── bbc_news_cleaned.csv
│   └── bbc_news_topics.csv
│
├── models/
│   ├── classifier.pkl
│   ├── count_vectorizer.pkl
│   ├── lda.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── download_nltk.py
│   ├── eda.py
│   ├── eda_visualization.py
│   ├── entity_extraction.py
│   ├── news_classifier.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── summarizer.py
│   ├── tfidf_analysis.py
│   └── topic_modeling.py
│
├── .devcontainer/
│   └── devcontainer.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🖥️ Application Workflow

The Streamlit application follows this workflow:

```text
                 ┌───────────────────┐
                 │ User Input        │
                 │ Text / URL        │
                 └─────────┬─────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Article         │
                  │ Extraction      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Preprocessing   │
                  └────────┬────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         Classification    LDA       TF-IDF
              │            │            │
              ▼            ▼            ▼
          Category       Topics      Keywords
              │
              └────────────┬────────────┐
                           │            │
                           ▼            ▼
                      Sentiment     NER
                           │            │
                           └─────┬──────┘
                                 ▼
                          Streamlit UI
```

The application displays:

- Predicted category
- Classification confidence
- Sentiment
- Article summary
- Topic probabilities
- Category probabilities
- Top keywords
- Named entities
- Original article
- Cleaned article

---

## 🛠️ Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Interface | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Classification | Multinomial Naive Bayes |
| Feature Extraction | TF-IDF, Count Vectorizer |
| Topic Modeling | LDA |
| NLP | NLTK |
| Sentiment | TextBlob |
| Web Scraping | Requests, BeautifulSoup |
| Model Serialization | Joblib |
| Visualization | Matplotlib, Seaborn |
| Word Analysis | WordCloud |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Swaritdixit/topic-analysis.git
cd topic-analysis
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

The application downloads the required NLTK resources when it starts.

The project also contains:

```text
src/download_nltk.py
```

which can be used to prepare the required NLTK resources.

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will provide the local application URL in the terminal.

---

## 🔄 Retraining the Models

The repository includes the trained model artifacts, so the Streamlit application can use the existing models directly.

The individual training/analysis scripts are located in `src/`.

### Preprocessing

```bash
python src/preprocessing.py
```

Generates:

```text
data/bbc_news_cleaned.csv
```

### Train the Classifier

```bash
python src/news_classifier.py
```

Generates:

```text
models/classifier.pkl
models/tfidf_vectorizer.pkl
```

### Train the LDA Model

```bash
python src/topic_modeling.py
```

Generates:

```text
models/lda.pkl
models/count_vectorizer.pkl
data/bbc_news_topics.csv
```

---

## 📦 Saved Model Artifacts

The repository contains pre-trained artifacts so that the deployed application does not need to retrain the models for every request.

```text
models/
│
├── classifier.pkl
├── tfidf_vectorizer.pkl
├── lda.pkl
└── count_vectorizer.pkl
```

These artifacts are loaded by `app.py` using `joblib`.

---

## 📌 Key Implementation Details

### Classification

**Algorithm:** Multinomial Naive Bayes

**Features:** TF-IDF

**Vocabulary:** Maximum 5,000 features

**Train/Test Split:** 80/20

```text
Training Data
     │
     ▼
TF-IDF
     │
     ▼
80% Training ───────► Multinomial NB
                         │
20% Testing ────────────┘
                         │
                         ▼
                  Classification
```

---

### Topic Modeling

**Algorithm:** Latent Dirichlet Allocation

**Number of topics:** 5

**Features:** Count Vectorization

The model produces a probability distribution across the discovered topics for each article.

---

### Summarization

The summarizer is **extractive**, meaning it selects important sentences from the original article rather than generating new sentences.

This makes it lightweight and easy to run without a large language model.

---

### Sentiment Analysis

TextBlob calculates the polarity of the article.

```text
Polarity > 0.1       → Positive

Polarity < -0.1      → Negative

Otherwise            → Neutral
```

---

### Named Entity Recognition

The application uses NLTK POS tagging and named entity chunking to identify entities in the original article.

---

## 🌐 Deployment

The application is deployed using **Streamlit**.

### Live Application

🔗 **https://topic-analyser.streamlit.app/**

The deployed application loads the pre-trained models from the repository and performs inference on user-provided articles.

---

## 🔮 Future Improvements

- Transformer-based summarization
- BERT-based topic modeling
- Transformer-based text classification
- Semantic keyword extraction
- Improved named entity recognition
- Better article extraction for different news websites
- Multi-language news analysis
- Real-time news ingestion
- News recommendation system
- Model evaluation dashboard
- More robust preprocessing
- Automated model retraining pipeline

---

## 🎯 What I Learned

This project provided hands-on experience with an end-to-end NLP workflow:

- Exploratory Data Analysis
- Text preprocessing
- Tokenization
- Stopword removal
- Lemmatization
- TF-IDF feature engineering
- Naive Bayes classification
- Model evaluation
- LDA topic modeling
- Extractive summarization
- Sentiment analysis
- Named entity recognition
- Web scraping
- Model serialization with Joblib
- Building an interactive Streamlit application
- Deploying an ML/NLP application

---

## 👨‍💻 Author

**Swarit Dixit**

B.Tech Electronics & Communication Engineering  
IIT Bhilai

- 💻 GitHub: https://github.com/Swaritdixit
- 💼 LinkedIn: https://www.linkedin.com/in/swarit-dixit-b907b8309/
