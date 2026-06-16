import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("data/bbc_news_cleaned.csv", sep=None,
    engine="python");
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df["clean_content"])
print(tfidf_matrix.shape)
feature_names=vectorizer.get_feature_names_out()
print(feature_names[:20])
scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
top_indices=scores.argsort()[-20:][::-1]

for idx in top_indices:  print(feature_names[idx], round(scores[idx], 4))
top_words=[feature_names[i] for i in top_indices]
top_scores=[scores[i] for i in top_indices]
plt.figure(figsize=(10,6))
plt.barh(top_words[::-1],top_scores[::-1])
plt.title("Top TF-IDF Words")
plt.xlabel("Average TF-IDF Score")
plt.tight_layout()
plt.show()
categories = df["category"].unique()

for category in categories:

    subset = df[
        df["category"] == category
    ]

    matrix = vectorizer.fit_transform(
        subset["clean_content"]
    )

    features = vectorizer.get_feature_names_out()

    scores = (
        matrix.mean(axis=0)
        .A1
    )

    top_indices = scores.argsort()[-10:][::-1]

    print("\n")
    print("="*50)
    print(category.upper())
    print("="*50)

    for idx in top_indices:
        print(features[idx])