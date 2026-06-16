import pandas as pd
df=pd.read_csv("data/bbc-news-data.csv", sep=None,
    engine="python");
print(df.head())
print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCategories:")
print(df["category"].value_counts())
