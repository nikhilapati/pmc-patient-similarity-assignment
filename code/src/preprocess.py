"""Clean the text: lowercase, remove punctuation, remove stop words, lemmatize."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nltk_resources():
    """Download NLTK's word lists only if they are not already on the computer."""
    for resource in ["stopwords", "wordnet"]:
        already_installed = False
        for path in [f"corpora/{resource}", f"corpora/{resource}.zip"]:
            try:
                nltk.data.find(path)
                already_installed = True
                break
            except LookupError:
                pass
        if not already_installed:
            nltk.download(resource, quiet=True)


def clean_text(text, stop_words, lemmatizer, min_token_length=2):
    """Clean one document."""
    text = text.lower()                                  # 1. standardization
    text = re.sub(r"[^a-z\s]", " ", text)                # 2. remove punctuation + numbers
    tokens = text.split()                                # 3. split into words
    tokens = [t for t in tokens
              if t not in stop_words and len(t) >= min_token_length]  # 4. drop stop words
    tokens = [lemmatizer.lemmatize(t) for t in tokens]   # 5. lemmatize
    return " ".join(tokens)


def preprocess_corpus(texts, min_token_length=2):
    """Clean a whole list of documents."""
    download_nltk_resources()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return [clean_text(t, stop_words, lemmatizer, min_token_length) for t in texts]