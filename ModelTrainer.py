import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from collections import Counter
from Helper import Helper
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import spacy
from nltk.corpus import words as nltk_words


class ModelTrainer:
    def __init__(self):
        self.curse_word_model = None
        self.slang_word_model = None
        self.name_model = None
        self.nlp = spacy.load('en_core_web_sm')
        self.english_words = set(nltk_words.words())

    def train_word2vec_model(self, all_lyrics):
        tokenized_lyrics = [lyric.split() for lyric in all_lyrics]
        return Word2Vec(sentences=tokenized_lyrics, vector_size=100, window=5, min_count=1, workers=4)

    def train_models(self, curse_words, slang_words, names, all_lyrics):
        # Train Word2Vec model on all lyrics
        word2vec_model = self.train_word2vec_model(all_lyrics)

        # Train models for curse words, slang, and names using word embeddings
        self.curse_word_model = self.train_model(curse_words, word2vec_model)
        self.slang_word_model = self.train_model(slang_words, word2vec_model)
        self.name_model = self.train_model(names, word2vec_model)

    def train_model(self, words, word2vec_model):
        # Get the word vectors for the given words
        word_vectors = []
        labels = []
        for word in words:
            if word in word2vec_model.wv:
                word_vectors.append(word2vec_model.wv[word])
                labels.append(1)

        # Create negative examples using random words from the vocabulary
        vocab = list(word2vec_model.wv.key_to_index.keys())
        negative_examples = [word2vec_model.wv[vocab[i]] for i in range(len(words))]
        word_vectors.extend(negative_examples)
        labels.extend([0] * len(negative_examples))

        X_train, X_test, y_train, y_test = train_test_split(word_vectors, labels, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        return model

    def predict_curse_words(self, text, word2vec_model):
        return self.predict_words(text, self.curse_word_model, word2vec_model)

    def predict_slang_words(self, text):
        words = text.split()
        slang_words = []
        for word in words:
            # Remove punctuation and check if the word ends with "'n" or is listed in Helper.Slang
            cleaned_word = word.strip(".,!?").lower()
            if (cleaned_word.endswith(
                    "in'") and cleaned_word not in self.english_words) or cleaned_word in Helper.Slang:
                slang_words.append(word)
        return dict(Counter(slang_words))

    def predict_names(self, text):
        name_words = [word for word in text.split() if word.lower() in Helper.Names]
        return dict(Counter(name_words))

    def predict_words(self, text, model, word2vec_model):
        if model:
            words = text.split()
            predictions = []
            for word in words:
                if word in word2vec_model.wv:
                    word_vector = word2vec_model.wv[word].reshape(1, -1)
                    prediction_prob = model.predict_proba(word_vector)[0][1]  # Get probability for the positive class
                    print(f"Word: {word}, Prediction Probability: {prediction_prob}")  # Log probability
                    if prediction_prob > 0.8:  # Adjust threshold as necessary
                        predictions.append(word)
            return dict(Counter(predictions))
        return {}

    @staticmethod
    def predict_words(text, model, word2vec_model):
        if model:
            words = text.split()
            predictions = []
            for word in words:
                if word in word2vec_model.wv:
                    word_vector = word2vec_model.wv[word].reshape(1, -1)
                    prediction = model.predict(word_vector)[0]
                    if prediction == 1:
                        predictions.append(word)
            return dict(Counter(predictions))
        return {}


