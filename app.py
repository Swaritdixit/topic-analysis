import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import requests

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.summarizer import generate_summary
from src.sentiment import get_sentiment
from src.entity_extraction import extract_entities

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("maxent_ne_chunker", quiet=True)
nltk.download("words", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

classifier = joblib.load(
    "models/classifier.pkl"
)

tfidf_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

lda = joblib.load(
    "models/lda.pkl"
)

count_vectorizer = joblib.load(
    "models/count_vectorizer.pkl"
)

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()

topic_names = {
    0: "Business",
    1: "Sports",
    2: "Politics",
    3: "Technology",
    4: "Entertainment"
}

def extract_article(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        article_text = " ".join(
            p.get_text(strip=True)
            for p in paragraphs
        )

        return article_text

    except Exception:

        return None

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)

st.set_page_config(
    page_title="Topic Analyzer",
    layout="wide"
)

st.title(
    "News Intelligence Platform"
)

input_type = st.radio(
    "Choose Input Type",
    [
        "Article Text",
        "Article URL"
    ]
)

article = ""

if input_type == "Article Text":

    article = st.text_area(
        "Paste a news article",
        height=300
    )

else:

    article_url = st.text_input(
        "Paste article URL"
    )

if st.button("Analyze"):

    if input_type == "Article URL":

        if not article_url.strip():

            st.warning(
                "Please enter a URL."
            )

            st.stop()

        article = extract_article(
            article_url
        )

        if not article:

            st.error(
                "Could not extract article."
            )

            st.stop()

    else:

        if not article.strip():

            st.warning(
                "Please enter an article."
            )

            st.stop()

    clean_article = preprocess_text(
        article
    )

    article_tfidf = (
        tfidf_vectorizer.transform(
            [clean_article]
        )
    )

    predicted_category = (
        classifier.predict(
            article_tfidf
        )[0]
    )

    probabilities = (
        classifier.predict_proba(
            article_tfidf
        )[0]
    )

    confidence = (
        probabilities.max()
        * 100
    )

    article_dtm = (
        count_vectorizer.transform(
            [clean_article]
        )
    )

    topic_probs = (
        lda.transform(
            article_dtm
        )[0]
    )

    predicted_topic = (
        topic_probs.argmax()
    )

    sentiment = get_sentiment(
        article
    )

    summary = generate_summary(
        article
    )

    entities = extract_entities(
        article
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Category",
            predicted_category
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    with col3:

        st.metric(
            "Sentiment",
            sentiment
        )

    st.subheader(
        "Article Summary"
    )

    st.write(summary)

    st.subheader(
        "Topic Probabilities"
    )

    topic_df = pd.DataFrame({
        "Topic": [
            topic_names.get(
                i,
                f"Topic {i}"
            )
            for i in range(
                len(topic_probs)
            )
        ],
        "Probability": topic_probs
    })

    st.bar_chart(
        topic_df.set_index(
            "Topic"
        )
    )

    st.subheader(
        "Category Probabilities"
    )

    category_df = pd.DataFrame({
        "Category":
        classifier.classes_,
        "Probability":
        probabilities
    })

    st.bar_chart(
        category_df.set_index(
            "Category"
        )
    )

    feature_names = (
        tfidf_vectorizer
        .get_feature_names_out()
    )

    row = article_tfidf.toarray()[0]

    top_indices = (
        row.argsort()
        [-10:][::-1]
    )

    keywords = [
        feature_names[i]
        for i in top_indices
        if row[i] > 0
    ]

    st.subheader(
        "Top Keywords"
    )

    st.write(
        ", ".join(keywords)
    )

    st.subheader(
        "Named Entities"
    )

    if entities:

        entity_df = pd.DataFrame(
            entities,
            columns=[
                "Entity",
                "Type"
            ]
        )

        st.dataframe(
            entity_df,
            use_container_width=True
        )

    else:

        st.write(
            "No entities found."
        )

    with st.expander(
        "Original Article"
    ):

        st.write(article)

    with st.expander(
        "Cleaned Article"
    ):

        st.write(clean_article)