import streamlit as st
import pandas as pd
import joblib
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

classifier = joblib.load("models/classifier.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
lda = joblib.load("models/lda.pkl")
count_vectorizer = joblib.load("models/count_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)

st.title("BBC News Topic Analyzer")

article = st.text_area(
    "Paste a news article",
    height=300
)

if st.button("Analyze"):

    clean_article = preprocess_text(article)

    article_tfidf = tfidf_vectorizer.transform(
        [clean_article]
    )

    predicted_category = classifier.predict(
        article_tfidf
    )[0]

    confidence = (
        classifier.predict_proba(article_tfidf)
        .max()
        * 100
    )

    article_dtm = count_vectorizer.transform(
        [clean_article]
    )

    topic_probs = lda.transform(
        article_dtm
    )[0]

    predicted_topic = topic_probs.argmax()

    st.subheader("Predicted Category")
    st.success(predicted_category)

    st.subheader("Confidence")
    st.write(f"{confidence:.2f}%")

    st.subheader("Predicted Topic")
    st.info(f"Topic {predicted_topic}")

    st.subheader("Topic Probabilities")
    st.bar_chart(topic_probs)

    feature_names = (
        tfidf_vectorizer
        .get_feature_names_out()
    )

    row = article_tfidf.toarray()[0]

    top_indices = row.argsort()[-10:][::-1]

    keywords = [
        feature_names[i]
        for i in top_indices
        if row[i] > 0
    ]

    st.subheader("Top Keywords")

    for word in keywords:
        st.write("•", word)