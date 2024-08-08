import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


class ModelTrainer:
    def __init__(self):
        self.curse_word_model = None
        self.slang_word_model = None
        self.name_model = None
        self.vectorizer = TfidfVectorizer()

    def train_models(self, curse_words, slang_words, names):
        self.curse_word_model = self.train_model(curse_words)
        self.slang_word_model = self.train_model(slang_words)
        self.name_model = self.train_model(names)

    def train_model(self, words):
        data = pd.DataFrame({
            'text': words + ["not_a_word"] * len(words),
            'label': [1] * len(words) + [0] * len(words)
        })
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
        model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))
        model.fit(X_train, y_train)
        return model

    def predict_curse_words(self, text):
        return self.predict_words(text, self.curse_word_model)

    def predict_slang_words(self, text):
        return self.predict_words(text, self.slang_word_model)

    def predict_names(self, text):
        return self.predict_words(text, self.name_model)

    def predict_words(self, text, model):
        if model:
            predictions = model.predict([text])
            return predictions[0]
        return None
