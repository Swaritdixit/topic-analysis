import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

df = pd.read_csv("data/bbc_news_cleaned.csv")

X = df["clean_content"]
y = df["category"]

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()

model.fit(X_train, y_train)
joblib.dump(model, "models/classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Classifier saved")
predictions = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)

print(
    classification_report(
        y_test,
        predictions
    )
)

print(confusion_matrix( y_test,predictions))
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test,predictions)

plt.figure(figsize=(8,6))

sns.heatmap(cm,annot=True,fmt="d",cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()