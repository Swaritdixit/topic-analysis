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

topic_names = {
    0: "Business",
    1: "Sports",
    2: "Politics",
    3: "Technology",
    4: "Entertainment"
}

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

st.set_page_config(
    page_title="BBC News Topic Analyzer",
    layout="wide"
)

st.title("BBC News Topic Analyzer")

article = st.text_area(
    "Paste a news article",
    height=300
)

if st.button("Analyze"):

    if article.strip() == "":
        st.warning("Please enter a news article.")
        st.stop()

    clean_article = preprocess_text(article)

    article_tfidf = tfidf_vectorizer.transform(
        [clean_article]
    )

    predicted_category = classifier.predict(
        article_tfidf
    )[0]

    probabilities = classifier.predict_proba(
        article_tfidf
    )[0]

    confidence = probabilities.max() * 100

    article_dtm = count_vectorizer.transform(
        [clean_article]
    )

    topic_probs = lda.transform(
        article_dtm
    )[0]

    predicted_topic = topic_probs.argmax()

    predicted_topic_name = topic_names.get(
        predicted_topic,
        f"Topic {predicted_topic}"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Category",
            predicted_category
        )

    with col2:
        st.metric(
            "Discovered Topic",
            predicted_topic_name
        )

    st.subheader("Classification Confidence")

    st.progress(int(confidence))

    st.write(
        f"{confidence:.2f}%"
    )

    st.subheader("Category Probabilities")

    category_df = pd.DataFrame({
        "Category": classifier.classes_,
        "Probability": probabilities
    })

    st.bar_chart(
        category_df.set_index("Category")
    )

    st.subheader("Topic Probabilities")

    topic_df = pd.DataFrame({
        "Topic": [
            topic_names.get(i, f"Topic {i}")
            for i in range(len(topic_probs))
        ],
        "Probability": topic_probs
    })

    st.bar_chart(
        topic_df.set_index("Topic")
    )

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

    keyword_df = pd.DataFrame({
        "Keyword": keywords
    })

    st.dataframe(
        keyword_df,
        use_container_width=True
    )

    st.subheader("Cleaned Text")

    st.text_area(
        "",
        clean_article,
        height=200
    )