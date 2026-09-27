# 📰 Topic Analyzer

An NLP-powered **News Intelligence Platform** built with Python, Machine Learning, NLP, and Streamlit.

Topic Analyzer analyzes news articles from either **pasted text or a URL** and provides:

* News category classification
* Classification probabilities and confidence
* Topic and subtopic discovery
* Topic-specific supporting sentences
* Extractive article summarization
* Sentiment analysis
* TF-IDF keyword extraction
* Named Entity Recognition
* Interactive analysis through Streamlit

**Live Demo:** https://topic-analyser.streamlit.app/

---

## Features

### 📰 News Classification

The application classifies an article into one of five categories using:

**TF-IDF → Multinomial Naive Bayes**

Supported categories:

* Business
* Politics
* Sports
* Technology
* Entertainment

The application also displays the probability assigned to each category using `predict_proba()`.

> The displayed "confidence" represents the highest predicted class probability. It should not be interpreted as model accuracy.

---

### 🔍 Topic and Subtopic Modeling

Topic discovery is performed using **Latent Dirichlet Allocation (LDA)** with five latent topics.

Unlike supervised classification, LDA does not use the predefined BBC categories. Instead, it discovers recurring word patterns and topic structures from the articles.

The application displays:

* Topic probability distribution
* Significant topics
* Top words associated with each topic
* Interpretable topic descriptions
* Topic-specific supporting sentences

A separate topic-specific TF-IDF vectorizer is used for the LDA pipeline.

---

### Topic-Specific Evidence

The application goes beyond displaying topic probabilities.

For each significant topic:

1. The article is split into sentences.
2. Each sentence is transformed using the topic TF-IDF vectorizer.
3. LDA estimates the topic distribution of each sentence.
4. Sentences are ranked according to their probability for the selected topic.
5. The strongest supporting sentences are displayed.

This provides sentence-level evidence for why a topic is considered significant.

If no individual sentence crosses the configured threshold, the application indicates that the topic signal is distributed across the article rather than strongly concentrated in one sentence.

---

### ✂️ Extractive Article Summarization

The summarizer generates a lightweight **extractive summary** without using a transformer or generative language model.

The process is:

```text
Article
   │
   ▼
Sentence Splitting
   │
   ▼
Word Tokenization
   │
   ▼
Stopword Removal
   │
   ▼
Word Frequency Calculation
   │
   ▼
Sentence Scoring
   │
   ▼
Top Sentences
   │
   ▼
Extractive Summary
```

The highest-scoring sentences are selected and combined to form the final summary.

---

### Sentiment Analysis

Sentiment analysis is performed using **TextBlob**.

TextBlob calculates a polarity score, which is mapped to:

* Positive
* Negative
* Neutral

The application uses polarity thresholds to determine the final sentiment.

---

### 🔑 Keyword Extraction

The classification TF-IDF representation is also used to identify the most prominent terms in an article.

The application:

1. Transforms the article using the trained TF-IDF vectorizer.
2. Retrieves the feature weights.
3. Ranks the terms by their TF-IDF scores.
4. Displays the highest-scoring terms as keywords.

The classifier itself uses the complete TF-IDF feature vector; keyword extraction is a separate interpretation layer built on top of those feature weights.

---

### Named Entity Recognition

Named Entity Recognition identifies important entities appearing in the article.

The application primarily uses **spaCy**, with an **NLTK fallback**.

Detected entity types can include:

* Person
* Organization
* Location
* Geopolitical Entity
* Nationality / Religious / Political Group
* Facility
* Event

The detected entities are cleaned, deduplicated, and displayed in a structured format.

---

### 🌐 URL-Based Article Analysis

Users can analyze either:

* Pasted article text
* A news article URL

For URL input, the application uses `Requests` and `BeautifulSoup` to extract article paragraphs.

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
NLP Analysis Pipeline
```

The request uses a browser-like User-Agent and a timeout to improve compatibility with common news websites.

URL extraction may not work correctly for websites that:

* Render article content entirely with JavaScript
* Require authentication
* Use strong bot protection
* Place unrelated content inside `<p>` elements
* Require a subscription or paywall

---

## Dataset

The project uses the **BBC News Dataset** containing 2,225 news articles across five categories.

| Category      |  Articles |
| ------------- | --------: |
| Sport         |       511 |
| Business      |       510 |
| Politics      |       417 |
| Technology    |       401 |
| Entertainment |       386 |
| **Total**     | **2,225** |

Each article contains information such as:

* Category
* Filename
* Title
* Content

---

## 🧠 Machine Learning Pipeline

The overall system combines supervised learning, unsupervised topic modeling, and additional NLP analysis.

```text
                         BBC News Dataset
                                │
                                ▼
                     Exploratory Data Analysis
                                │
                                ▼
                       Text Preprocessing
                                │
                ┌───────────────┴────────────────┐
                │                                │
                ▼                                ▼
        Classification Pipeline            Topic Pipeline
                │                                │
                ▼                                ▼
             TF-IDF                    Topic TF-IDF
                │                                │
                ▼                                ▼
     Multinomial Naive Bayes                    LDA
                │                                │
                ▼                                ▼
     News Category Prediction          Topic Distribution
                │                                │
                │                         Sentence-Level
                │                         Topic Evidence
                │                                │
                └───────────────┬────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
               Summary     Sentiment       NER
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                         Streamlit Dashboard
```

---

## Text Preprocessing

The preprocessing pipeline performs the following operations:

### 1. Lowercasing

Converts text to lowercase.

```text
"Technology News"
        ↓
"technology news"
```

### 2. URL Removal

URLs are removed using regular expressions.

### 3. Number Removal

Numeric characters are removed from the text.

### 4. Non-Alphabetic Character Removal

Non-alphabetic characters are removed before tokenization.

### 5. Tokenization

The article is split into individual words using NLTK.

### 6. Stopword Removal

Common English stopwords are removed.

Examples:

```text
the
is
and
of
to
```

### 7. Lemmatization

Words are reduced to their base form using NLTK's `WordNetLemmatizer`.

For example:

```text
running → running
cars    → car
better  → better
```

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
Numerical Feature Vector
      │
      ▼
Multinomial Naive Bayes
      │
      ├── predict()
      │
      └── predict_proba()
              │
              ▼
     Category + Probabilities
```

### What TF-IDF does

TF-IDF converts text into numerical features based on how important a word is within a document and across the collection of documents.

Conceptually:

```text
TF-IDF = Term Frequency × Inverse Document Frequency
```

A word receives a higher weight when it is important within a document but is not equally common across the entire dataset.

### Why the vectorizer is saved

The trained TF-IDF vectorizer stores:

* Vocabulary
* Feature ordering
* IDF values
* Feature configuration

The same vectorizer must therefore be used during inference.

The application uses:

```python
vectorizer.transform(text)
```

rather than fitting a new vectorizer for every article.

### Multinomial Naive Bayes

Multinomial Naive Bayes is used as the supervised classifier because it is:

* Fast
* Lightweight
* Well suited to text classification
* Simple to train and deploy

The model learns the relationship between text features and the five known news categories.

---

## 🧩 LDA Topic Modeling

Topic discovery uses **Latent Dirichlet Allocation (LDA)**.

The project uses:

```text
Number of Topics = 5
```

The topic pipeline uses a separate TF-IDF representation from the classification pipeline.

```text
Cleaned Articles
       │
       ▼
Topic TF-IDF Vectorizer
       │
       ▼
Document-Term Representation
       │
       ▼
LDA
       │
       ▼
5 Latent Topic Distributions
```

### What LDA provides

For every article, LDA produces a probability distribution across the five discovered topics.

For example:

```text
Topic 1 → 0.08
Topic 2 → 0.52
Topic 3 → 0.11
Topic 4 → 0.21
Topic 5 → 0.08
```

The topic numbers themselves do not have predefined meanings.

The application interprets them using the highest-weighted words associated with each topic.

For example:

```text
Topic 1
├── election
├── government
├── minister
├── parliament
└── vote
```

A human can interpret this topic as being related to politics, but LDA itself only discovers the underlying word patterns.

---

## Sentence-Level Topic Analysis

The application performs an additional analysis to determine which sentences support each significant topic.

```text
Article
   │
   ▼
Split into Sentences
   │
   ▼
Topic TF-IDF Transformation
   │
   ▼
LDA Topic Probabilities
   │
   ▼
Rank Sentences by Topic Score
   │
   ▼
Select Strong Supporting Sentences
```

Up to two strong sentences can be displayed for each significant topic.

This makes the topic analysis more interpretable because the user can see actual evidence from the article rather than only a numerical topic distribution.

---

## 📊 Exploratory Data Analysis

The project includes exploratory analysis of the BBC News dataset.

The EDA examines:

* Dataset dimensions
* Column information
* Missing values
* Category distribution
* Article length
* Average article length by category
* Word frequency

Visualizations include:

* Category distribution
* Article length distribution
* Average article length by category
* Word cloud

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
│   ├── lda.pkl
│   ├── tfidf_vectorizer.pkl
│   └── topic_tfidf_vectorizer.pkl
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
                    ┌──────────────────────┐
                    │      User Input      │
                    │      Text / URL      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Article Extraction   │
                    │ if URL is provided   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Preprocessing   │
                    └──────────┬───────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
        Classification    Topic Modeling    NLP Analysis
             │                 │                 │
             ▼                 ▼          ┌──────┼──────┐
        TF-IDF + NB          LDA          │      │      │
             │                 │          ▼      ▼      ▼
             ▼                 ▼       Summary Sentiment NER
        Category +        Topics          │      │      │
        Probabilities         │           └──────┼──────┘
             │                 │                  │
             └─────────────────┼──────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    └──────────────────────┘
```

The dashboard displays:

* Predicted category
* Category probabilities
* Classification confidence
* Sentiment
* Extractive summary
* Topic distribution
* Topic-specific supporting sentences
* Top keywords
* Named entities
* Original article
* Cleaned article

---

## 🛠️ Technology Stack

| Area                | Technologies            |
| ------------------- | ----------------------- |
| Language            | Python                  |
| Interface           | Streamlit               |
| Data Processing     | Pandas, NumPy           |
| Machine Learning    | Scikit-learn            |
| Classification      | Multinomial Naive Bayes |
| Feature Extraction  | TF-IDF                  |
| Topic Modeling      | LDA                     |
| NLP                 | NLTK, spaCy             |
| Sentiment Analysis  | TextBlob                |
| Web Scraping        | Requests, BeautifulSoup |
| Model Serialization | Joblib                  |
| Visualization       | Matplotlib, Seaborn     |
| Word Analysis       | WordCloud               |

---

## Getting Started

### Prerequisites

* Python 3.x
* pip

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

### 4. Download NLP Resources

The application handles the required NLTK resources during startup.

The repository also contains:

```text
src/download_nltk.py
```

which can be used to prepare the required NLTK resources.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will be available at the local Streamlit URL shown in the terminal.

---

## 🔄 Retraining the Models

The repository contains pre-trained model artifacts, so the deployed application does not retrain models for every request.

The training and analysis scripts are located inside `src/`.

### Preprocessing

```bash
python src/preprocessing.py
```

Generates the cleaned dataset:

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

### Train the Topic Model

```bash
python src/topic_modeling.py
```

Generates the LDA model and topic-specific vectorizer used by the topic pipeline.

```text
models/lda.pkl
models/topic_tfidf_vectorizer.pkl
```

---

## 📦 Saved Model Artifacts

The application uses serialized models and vectorizers stored in the `models/` directory.

| File                         | Purpose                                                     |
| ---------------------------- | ----------------------------------------------------------- |
| `classifier.pkl`             | Trained Multinomial Naive Bayes classifier                  |
| `tfidf_vectorizer.pkl`       | TF-IDF vectorizer for classification and keyword extraction |
| `lda.pkl`                    | Trained LDA topic model                                     |
| `topic_tfidf_vectorizer.pkl` | TF-IDF vectorizer used by the topic pipeline                |

Using saved artifacts allows the deployed application to perform inference without retraining the models for each request.

---

## Deployment

The application is deployed using **Streamlit Community Cloud**.

Live application:

https://topic-analyser.streamlit.app/

The deployment uses the repository's:

```text
app.py
requirements.txt
models/
src/
```

The pre-trained artifacts are loaded when the application starts.

---

## Limitations

The current system uses lightweight classical NLP techniques, so there are several areas that can be improved.

### Article Extraction

The URL extractor relies on HTML paragraph elements and may not work reliably with JavaScript-heavy websites or protected pages.

### Classification

The classifier is trained on the BBC News dataset, so its performance may differ on articles from other sources or domains.

### Topic Interpretation

LDA produces latent topics rather than human-readable topic names. Topic labels are interpreted using their highest-weighted words.

### Summarization

The summarizer is extractive and frequency-based. It does not generate new sentences or understand article context like modern transformer-based summarizers.

### Sentiment

TextBlob provides a general polarity score and may struggle with sarcasm, complex context, or domain-specific language.

### Preprocessing

Classical preprocessing techniques may remove contextual information that modern language models can preserve.

---

## Future Improvements

Potential improvements include:

* Transformer-based text classification
* Transformer-based abstractive summarization
* Semantic keyword extraction
* Better article-content extraction
* Multilingual news analysis
* Real-time news ingestion
* Topic naming using language models
* Improved topic coherence evaluation
* Model performance dashboard
* Automated model retraining
* More robust preprocessing and evaluation
* Article recommendation based on semantic similarity

---

## What I Learned

This project helped me understand how multiple NLP techniques can be combined into a single end-to-end application.

Key areas explored include:

* Text preprocessing
* Tokenization and lemmatization
* TF-IDF feature engineering
* Multinomial Naive Bayes
* Probability-based classification
* Unsupervised topic modeling with LDA
* Sentence-level topic analysis
* Extractive summarization
* Sentiment analysis
* Named Entity Recognition
* Web scraping
* Model serialization with Joblib
* Streamlit application development
* Deployment of ML applications

The project also demonstrates how traditional machine learning and NLP techniques can be combined to build an interpretable news analysis platform.

---

## Author

**Swarit Dixit**

B.Tech ECE, IIT Bhilai

* GitHub: https://github.com/Swaritdixit
* LinkedIn: https://www.linkedin.com/in/swarit-dixit-b907b8309/

---

## License

This project is intended for educational and portfolio purposes.
