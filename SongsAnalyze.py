import spacy
import nltk
import pytextrank
from collections import Counter
from nltk.tokenize import word_tokenize
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import numpy as np
import pandas as pd
from Helper import Helper
from ModelTrainer import ModelTrainer



class SongsAnalyze:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe("textrank")  # Add pytextrank to the SpaCy pipeline
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.real_words = set(nltk.corpus.words.words())
        self.model_trainer = ModelTrainer()  # Initialize ModelTrainer

    def analyze_song(self, song, all_lyrics):
        """
        Analyzes a single song and returns the analysis results.
        """
        song = Helper.preprocess_lyrics(song)  # Preprocess the lyrics
        word2vec_model = self.model_trainer.train_word2vec_model(all_lyrics)  # Get the trained Word2Vec model
        analysis_results = {
            "word_frequency": self.get_word_frequency(song),
            "named_entities": self.get_named_entities(song),
            "dependency_parse": self.get_dependency_parse(song),
            "topic_modeling": self.get_topic_modeling([song]),  # Perform topic modeling on the single song
            "non_real_words_freq": self.get_non_real_words_frequency(song),
            "curse_words_freq": self.get_curse_words_frequency(song),
            "sentiment_analysis": self.get_sentiment_analysis(song),  # Sentiment analysis of the song
            "predicted_curse_words": self.model_trainer.predict_curse_words(song),  # Predicted curse words
            "predicted_slang_words": self.model_trainer.predict_slang_words(song),  # Predicted slang words
            "predicted_names": self.model_trainer.predict_names(song)  # Predicted names
        }
        return analysis_results

    def get_word_frequency(self, text):
        """
        Calculates the frequency of each word in the text.
        Words are tokenized, converted to lowercase, and non-alphanumeric tokens are excluded.
        """
        words = [word for word in word_tokenize(text.lower()) if word.isalnum()]
        return dict(Counter(words))

    def get_named_entities(self, text):
        """
        Extracts named entities and their frequencies from the text.
        Named entities include proper names such as people, organizations, and locations.
        """
        doc = self.nlp(text)
        entities = [ent.label_ for ent in doc.ents]
        return dict(Counter(entities))

    def get_dependency_parse(self, text):
        """
        Calculates the frequency of dependency parse tags in the text.
        Dependency parsing provides information about the grammatical structure of sentences.
        """
        doc = self.nlp(text)
        dependencies = [token.dep_ for token in doc]
        return dict(Counter(dependencies))

    def get_topic_modeling(self, texts):
        """
        Performs topic modeling on the texts using Latent Dirichlet Allocation (LDA).
        Identifies common topics discussed in the texts.
        """
        tokens = [[word for word in word_tokenize(text.lower()) if word.isalnum() and word not in self.stop_words] for
                  text in texts]
        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(text) for text in tokens]
        lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
        topics = lda_model.print_topics(num_words=10)
        return topics

    def get_non_real_words_frequency(self, text):
        """
        Calculates the frequency of non-real words (words not found in the English dictionary) in the text.
        """
        words = [word for word in word_tokenize(text.lower()) if word.isalnum()]
        non_real_words = [word for word in words if word not in self.real_words]
        return dict(Counter(non_real_words))

    def get_curse_words_frequency(self, text):
        """
        Calculates the frequency of curse words in the text.
        """
        words = [word for word in word_tokenize(text.lower()) if word.isalnum()]
        curse_words = [word for word in words if word in Helper.CURSE_WORDS]
        return dict(Counter(curse_words))

    def get_sentiment_analysis(self, text):
        """
        Performs sentiment analysis on the text and returns the negative words and average number of negative words per sentence.
        """
        blob = TextBlob(text)
        negative_words = []
        for sentence in blob.sentences:
            if sentence.sentiment.polarity < 0:
                negative_words.extend([word for word in sentence.words if
                                       word.lower() in self.real_words and word.lower() not in self.stop_words])
        avg_negative_words_per_sentence = len(negative_words) / len(blob.sentences) if blob.sentences else 0
        return {
            "negative_words": dict(Counter(negative_words)),
            "average_negative_words_per_sentence": avg_negative_words_per_sentence
        }

    def get_predicted_curse_words(self, text):
        """
        Identifies curse words based on a predefined list in the Helper class.
        """
        words = [word for word in text.split() if word.lower() in Helper.CURSE_WORDS]
        return dict(Counter(words))

    def get_predicted_slang_words(self, text):
        """
        Uses the trained model to predict slang words in the text.
        """
        predictions = [word for word in text.split() if self.model_trainer.predict_slang_words(word) == 1]
        return dict(Counter(predictions))

    def get_predicted_names(self, text):
        """
        Uses the trained model to predict names in the text.
        """
        predictions = [word for word in text.split() if self.model_trainer.predict_names(word) == 1]
        return dict(Counter(predictions))


def load_lyrics_from_file(file_path):
    """
    Loads lyrics from a single CSV file.
    Assumes each CSV file has columns 'Song Title', 'Lyrics', and 'Album'.
    """
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df
