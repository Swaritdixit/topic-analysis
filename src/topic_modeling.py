import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv("data/bbc_news_cleaned.csv")

vectorizer = CountVectorizer( max_df=0.95,min_df=2,stop_words="english")

dtm = vectorizer.fit_transform(df["clean_content"])

lda = LatentDirichletAllocation(n_components=5,random_state=42)

lda.fit(dtm)
joblib.dump(lda, "models/lda.pkl")
joblib.dump(vectorizer, "models/count_vectorizer.pkl")

print("LDA model saved")
feature_names = vectorizer.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):

    print("\n")
    print("=" * 60)
    print(f"TOPIC {topic_idx + 1}")
    print("=" * 60)

    top_words = [feature_names[i]
        for i in topic.argsort()[-10:][::-1] ]

    print(" ".join(top_words))

topic_results = lda.transform(dtm)

df["topic"] = topic_results.argmax(axis=1)

comparison = pd.crosstab( df["category"],df["topic"])

fig = plt.figure(figsize=(18, 12))

ax1 = plt.subplot(2, 2, 1)

df["topic"].value_counts().sort_index().plot(
    kind="bar",
    ax=ax1
)

ax1.set_title("Topic Distribution")
ax1.set_xlabel("Topic")
ax1.set_ylabel("Articles")

ax2 = plt.subplot(2, 2, 2)

comparison.plot(kind="bar",ax=ax2)

ax2.set_title("BBC Categories vs Topics")
ax2.set_xlabel("Category")
ax2.set_ylabel("Articles")

ax3 = plt.subplot(2, 2, 3)

sns.heatmap(comparison,annot=True,fmt="d",cmap="Blues",ax=ax3)

ax3.set_title("Category Topic Heatmap")

ax4 = plt.subplot(2, 2, 4)

topic_percentages = (df["topic"].value_counts(normalize=True).sort_index()* 100)

ax4.pie(topic_percentages,labels=topic_percentages.index,autopct="%1.1f%%"
)

ax4.set_title("Topic Percentage Distribution")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10)
)

axes = axes.flatten()

for topic_idx, topic in enumerate(lda.components_):

    top_indices = topic.argsort()[-10:][::-1]

    words = [
        feature_names[i]
        for i in top_indices
    ]

    scores = [
        topic[i]
        for i in top_indices
    ]

    axes[topic_idx].barh(
        words[::-1],
        scores[::-1]
    )

    axes[topic_idx].set_title(
        f"Topic {topic_idx + 1}"
    )

if len(axes) > 5:
    fig.delaxes(axes[5])

plt.tight_layout()
plt.show()

df.to_csv( "data/bbc_news_topics.csv", index=False)

print(comparison)