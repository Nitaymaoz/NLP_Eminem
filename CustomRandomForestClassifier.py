import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
import joblib
import heapq


class CustomRandomForestClassifier:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.vectorizer = None
        self.model = None

    def load_and_preprocess_data(self, first_period_path, second_period_path):
        first_period_df = pd.read_csv(first_period_path)
        second_period_df = pd.read_csv(second_period_path)

        first_period_df['Period'] = 'First'
        second_period_df['Period'] = 'Second'

        data = pd.concat([first_period_df, second_period_df])

        def clean_lyrics(lyrics):
            lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove text in brackets
            lyrics = re.sub(r'\s+', ' ', lyrics)  # Replace multiple spaces with a single space
            lyrics = re.sub(r'[^\w\s]', '', lyrics)  # Remove punctuation
            return lyrics.lower()  # Convert to lowercase

        data['Cleaned_Lyrics'] = data['Lyrics'].apply(clean_lyrics)
        self.data = data

    def vectorize_text(self, max_features):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        X = self.vectorizer.fit_transform(self.data['Cleaned_Lyrics'])
        y = self.data['Period']
        return X, y

    def find_best_max_features(self):
        max_heap = []
        for max_features in range(1, 5001, 100):  # Adjust step size as needed
            X, y = self.vectorize_text(max_features)
            model = RandomForestClassifier(random_state=self.random_seed)
            scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
            accuracy = scores.mean()
            heapq.heappush(max_heap, (-accuracy, max_features))

        best_accuracy, best_max_features = heapq.heappop(max_heap)
        return -best_accuracy, best_max_features

    def train_final_model(self, max_features):
        X, y = self.vectorize_text(max_features)
        self.model = RandomForestClassifier(random_state=self.random_seed)
        self.model.fit(X, y)
        y_pred = cross_val_predict(self.model, X, y, cv=5)  # 5-fold cross-validation
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        return accuracy, report

    def save_model(self, model_path, vectorizer_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
