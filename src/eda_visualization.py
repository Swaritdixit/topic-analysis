import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
df=pd.read_csv("data/bbc-news-data.csv", sep=None,
    engine="python")
category_counts=df["category"].value_counts()

df["article_length"]=df["content"].apply(len)

avg_length=(df.groupby("category")["article_length"].mean().sort_values(ascending=False))
all_text = " ".join(df["content"])

wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white"
).generate(all_text)

plt.figure(figsize=(8,5))
category_counts.plot(kind="bar")
plt.title("Number of Articles per Category")

plt.figure(figsize=(8,5))
plt.hist(df["article_length"], bins=30)
plt.title("Distribution of Article Length")

plt.figure(figsize=(8,5))
avg_length.plot(kind="bar")
plt.title("Average Article Length by Category")

plt.figure(figsize=(12,6))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Most Frequent Words in BBC News Dataset")

plt.show()


