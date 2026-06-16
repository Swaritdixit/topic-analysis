import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
df=pd.read_csv("data/bbc-news-data.csv", sep=None,
    engine="python")
stop_words=set(stopwords.words("english"))
lemmatizer=WordNetLemmatizer()
def preprocess_text(text):
    text=text.lower()
    text=re.sub(r'http\S+','',text)
    text=re.sub(r'\d+','',text)
    tokens=word_tokenize(text)
    tokens=[word for word in tokens 
            if word not in stop_words]
    tokens=[lemmatizer.lemmatize(word)
            for word in tokens]
    return " ".join(tokens)

df["clean_content"]=df["content"].apply(preprocess_text)
print("Original Article:\n")
print(df["content"].iloc[0][:500])

print("\n\ncleaned article:\n")
print(df["clean_content"].iloc[0][:500])

df.to_csv( "data/bbc_news_cleaned.csv",
    index=False)