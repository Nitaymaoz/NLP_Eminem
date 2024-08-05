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
import os
import pandas as pd

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')

# List of common curse words
CURSE_WORDS = {'fuck', 'shit', 'bitch', 'ass', 'damn', 'bastard', 'hell', 'dick', 'piss', 'crap', 'prick', 'cock',
               'pussy', 'slut', 'douche', 'cunt', 'motherfucker', 'fucking','ballsack','nuts'}

class SongsAnalyze:

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe("textrank")  # Add pytextrank to the SpaCy pipeline
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.real_words = set(nltk.corpus.words.words())

    def analyze_song(self, song):
        """
        Analyzes a single song and returns the analysis results.
        """
        analysis_results = {
            "word_frequency": self.get_word_frequency(song),
            "named_entities": self.get_named_entities(song),
            "dependency_parse": self.get_dependency_parse(song),
            "text_classification": None,  # Placeholder for future text classification
            "text_similarity_average": None,  # Not applicable for individual songs
            "topic_modeling": self.get_topic_modeling([song]),  # Perform topic modeling on the single song
            "non_real_words_freq": self.get_non_real_words_frequency(song),
            "curse_words_freq": self.get_curse_words_frequency(song),
            "sentiment_analysis": self.get_sentiment_analysis(song)  # Sentiment analysis of the song
        }
        return analysis_results

    def analyze_album(self, album, song_titles):
        """
        Analyzes an album by analyzing each song individually.
        Returns a list of analysis results for each song.
        """
        album_analysis = []
        for title, song in zip(song_titles, album):
            song_analysis = self.analyze_song(song)
            album_analysis.append((title, song_analysis))
        return album_analysis

    def get_pos_tags_frequency(self, text):
        """
        Calculates the frequency of Part-of-Speech (POS) tags in the text.
        POS tags indicate the grammatical role of each word (e.g., noun, verb, adjective).
        """
        doc = self.nlp(text)
        pos_tags = [token.pos_ for token in doc]
        return dict(Counter(pos_tags))

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

    def get_text_similarity_average(self, texts):
        """
        Computes the average text similarity between all pairs of texts.
        TF-IDF vectors are used to measure similarity.
        """
        doc_vectors = self.vectorizer.fit_transform(texts)
        similarity_matrix = doc_vectors * doc_vectors.T
        similarities = similarity_matrix.A[np.triu_indices_from(similarity_matrix.A, k=1)]
        return similarities.mean() if similarities.size else 0.0

    def get_text_lemmatization_frequency(self, text):
        """
        Calculates the frequency of lemmatized words in the text.
        Lemmatization reduces words to their base or root form.
        """
        lemmas = [self.lemmatizer.lemmatize(word) for word in word_tokenize(text.lower()) if word.isalnum()]
        return dict(Counter(lemmas))

    def get_text_summarization(self, text):
        """
        Generates a summary of the text.
        This function uses spaCy's TextRank algorithm to summarize the text.
        """
        doc = self.nlp(text)
        return ' '.join([phrase.text for phrase in doc._.textrank.summary(limit_phrases=15)])

    def get_topic_modeling(self, texts):
        """
        Performs topic modeling on the texts using Latent Dirichlet Allocation (LDA).
        Identifies common topics discussed in the texts.
        """
        tokens = [[word for word in word_tokenize(text.lower()) if word.isalnum() and word not in self.stop_words] for text in texts]
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
        curse_words = [word for word in words if word in CURSE_WORDS]
        return dict(Counter(curse_words))

    def get_sentiment_analysis(self, text):
        """
        Performs sentiment analysis on the text and returns the negative words and average number of negative words per sentence.
        """
        blob = TextBlob(text)
        negative_words = []
        for sentence in blob.sentences:
            if sentence.sentiment.polarity < 0:
                negative_words.extend([word for word in sentence.words if word.lower() in self.real_words and word.lower() not in self.stop_words])
        avg_negative_words_per_sentence = len(negative_words) / len(blob.sentences) if blob.sentences else 0
        return {
            "negative_words": dict(Counter(negative_words)),
            "average_negative_words_per_sentence": avg_negative_words_per_sentence
        }

    # def vectorize_period(self, albums):
    #     """
    #     Vectorizes the analyses of all albums in a period.
    #     """
    #     album_vectors = []
    #     for album in albums:
    #         album_text = ' '.join(album)
    #         cleaned_text = ' '.join([word for word in album_text.split() if word not in self.stop_words])
    #         if cleaned_text:  # Ensure non-empty after removing stop words
    #             album_vector = self.vectorizer.fit_transform([cleaned_text])
    #             album_vectors.append(album_vector.toarray())
    #     if album_vectors:
    #         return np.mean(album_vectors, axis=0)
    #     return np.array([])


def load_lyrics_from_file(file_path):
    """
    Loads lyrics from a single CSV file.
    Assumes each CSV file has columns 'Song Title', 'Lyrics', and 'Album'.
    """
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    return df
