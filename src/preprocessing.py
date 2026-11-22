
import nltk
import numpy as np

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# Initialize Porter Stemmer
stemmer = PorterStemmer()


# Download required NLTK tokenizer resource if missing
def download_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


download_nltk_resources()


# Convert sentence into individual words
def tokenize(sentence):
    return word_tokenize(sentence.lower())


# Convert words into their root form
def stem(word):
    return stemmer.stem(word.lower())


# Remove punctuation and stem words
def preprocess(sentence):
    words = tokenize(sentence)

    processed_words = []

    for word in words:
        if word.isalnum():
            processed_words.append(stem(word))

    return processed_words


# Create Bag of Words representation
def bag_of_words(tokenized_sentence, all_words):
    sentence_words = [
        stem(word) for word in tokenized_sentence
        if word.isalnum()
    ]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, word in enumerate(all_words):
        if word in sentence_words:
            bag[index] = 1.0

    return bag


# Build vocabulary and intent tags from dataset
def build_vocabulary(intents):
    all_words = []
    tags = []

    for intent in intents["intents"]:
        tag = intent["tag"]

        if tag not in tags:
            tags.append(tag)

        for pattern in intent["patterns"]:
            words = preprocess(pattern)
            all_words.extend(words)

    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    return all_words, tags
