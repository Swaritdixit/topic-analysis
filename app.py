import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import requests

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from src.summarizer import generate_summary
from src.sentiment import get_sentiment
from src.entity_extraction import extract_entities

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)          
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

classifier = joblib.load(
    "models/classifier.pkl"
)

tfidf_vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


topic_vectorizer = joblib.load(
    "models/topic_tfidf_vectorizer.pkl"
)

lda = joblib.load(
    "models/lda.pkl"
)

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()

topic_feature_names = topic_vectorizer.get_feature_names_out()

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
            p.get_text(separator=" ", strip=True)
            for p in paragraphs
        )

     
        article_text = re.sub(r"\s+", " ", article_text).strip()

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

    # LDA runs on its own topic-specific TF-IDF vector (see model
    # loading above) -- separate from the classifier's TF-IDF vector.
    article_topic_tfidf = (
        topic_vectorizer.transform(
            [clean_article]
        )
    )

    topic_probs = (
        lda.transform(
            article_topic_tfidf
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
        "Subtopics Found in This Article (LDA on TF-IDF)"
    )

    st.caption(
        "Below, only the topics that actually show up in THIS article "
        "are listed as subtopics -- ranked by how much of the article "
        "they cover -- each with real example sentences pulled from "
        "the text, independent of the classifier's category below."
    )

    raw_sentences = sent_tokenize(article)


    sentence_records = []

    for raw_sentence in raw_sentences:

        clean_sentence = preprocess_text(raw_sentence)

        if not clean_sentence.strip():
            continue

        sentence_topic_tfidf = topic_vectorizer.transform(
            [clean_sentence]
        )

        if sentence_topic_tfidf.nnz == 0:
            continue

        sentence_topic_probs = lda.transform(
            sentence_topic_tfidf
        )[0]

        sentence_records.append(
            (raw_sentence.strip(), sentence_topic_probs)
        )


    SUBTOPIC_WEIGHT_THRESHOLD = 0.15
    EXAMPLES_PER_SUBTOPIC = 2

    MIN_EXAMPLE_SCORE = (1.0 / lda.n_components) * 1.7

    ranked_topics = sorted(
        range(lda.n_components),
        key=lambda i: topic_probs[i],
        reverse=True
    )

    subtopics = [
        i for i in ranked_topics
        if topic_probs[i] >= SUBTOPIC_WEIGHT_THRESHOLD
    ]

 
    if not subtopics:
        subtopics = ranked_topics[:2]

    for rank, i in enumerate(subtopics, start=1):

        article_weight_pct = topic_probs[i] * 100

        # Score every sentence specifically against THIS subtopic (not
        # its own argmax), so a sentence that's a strong secondary match
        # for this subtopic can still surface as an example.
        candidates = sorted(
            (
                (sentence, probs[i])
                for sentence, probs in sentence_records
                if probs[i] >= MIN_EXAMPLE_SCORE
            ),
            key=lambda pair: pair[1],
            reverse=True
        )

        examples = candidates[:EXAMPLES_PER_SUBTOPIC]

        st.markdown(
            f"**Subtopic {rank}** "
            f"— {article_weight_pct:.1f}% of this article"
        )

        if examples:
            for sentence, sentence_score in examples:
                st.write(f"> {sentence}")
        else:
            st.write(
                "_No single sentence strongly matches this subtopic on "
                "its own -- the signal is spread thinly across the "
                "whole article._"
            )

        st.write("")

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
