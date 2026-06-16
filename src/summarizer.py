import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter

def generate_summary(text, num_sentences=5):

    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    stop_words = set(stopwords.words("english"))

    words = word_tokenize(text.lower())

    words = [
        word
        for word in words
        if word.isalnum()
        and word not in stop_words
    ]

    word_freq = Counter(words)

    sentence_scores = {}

    for sentence in sentences:

        sentence_words = word_tokenize(
            sentence.lower()
        )

        score = sum(
            word_freq.get(word, 0)
            for word in sentence_words
        )

        sentence_scores[sentence] = score

    summary_sentences = sorted(
        sentence_scores,
        key=sentence_scores.get,
        reverse=True
    )[:num_sentences]

    return " ".join(summary_sentences)