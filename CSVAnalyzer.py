import pandas as pd
from collections import Counter
import numpy as np

class CSVAnalyzer:
    def __init__(self, file_paths, output_filenames):
        self.file_paths = file_paths
        self.output_filenames = output_filenames

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def process_files(self):
        for i, file_path in enumerate(self.file_paths):
            print(f"Processing file: {file_path}")
            df = self.load_data(file_path)
            processed_results = self.analyze_dataframe(df)
            self.save_to_csv(processed_results, self.output_filenames[i])

    def analyze_dataframe(self, df):
        results = {}

        # Average and Mean for specified columns
        columns_to_average = ['total_unique_words', 'total_words', 'total_names',
                              'total_slangs', 'total_curse_words']
        for col in columns_to_average:
            results[f'avg_{col}'] = df[col].mean()
            results[f'median_{col}'] = df[col].median()

        # Summing and Counting for specified columns
        columns_to_sum = ['predicted_curse_words', 'predicted_slang_words',
                          'predicted_names', 'word_frequency']

        for col in columns_to_sum:
            summed_col = Counter()
            for entry in df[col]:
                summed_col.update(eval(entry))  # Using eval to convert string to dict
            results[f'summed_{col}'] = dict(summed_col)

        # Found album and song name references
        results['found_albums_names_refs'] = self.find_references(df, 'found_albums_names_refs')
        results['found_songs_names_refs'] = self.find_references(df, 'found_songs_names_refs')

        # Sentiment analysis
        sentiment = self.calculate_average_sentiment(df)
        results.update(sentiment)

        # Named entities average
        named_entities_avg = self.calculate_average_named_entities(df)
        results['average_named_entities'] = named_entities_avg

        return results

    def find_references(self, df, column_name):
        all_references = set()
        for entry in df[column_name]:
            references = eval(entry)  # Using eval to convert string to set
            all_references.update(references)
        return list(all_references)

    def calculate_average_sentiment(self, df):
        sentiment_col = df['sentiment_analysis']
        sentiment_counter = Counter()

        for entry in sentiment_col:
            sentiment_dict = eval(entry)
            sentiment_counter.update(sentiment_dict['negative_words'])

        for word in sentiment_counter:
            sentiment_counter[word] /= len(sentiment_col)  # Averaging per word

        return {'average_sentiment': dict(sentiment_counter)}

    def calculate_average_named_entities(self, df):
        named_entities_col = df['named_entities']
        entity_counter = Counter()

        for entry in named_entities_col:
            entities_dict = eval(entry)
            entity_counter.update(entities_dict)

        for entity in entity_counter:
            entity_counter[entity] /= len(named_entities_col)  # Averaging per entity

        return dict(entity_counter)

    def save_to_csv(self, data, filename):
        df = pd.DataFrame([data])
        df.to_csv(filename, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    file_paths = ['song_analysis_first_period.csv', 'song_analysis_second_period.csv']
    output_filenames = ['summary_first_period.csv', 'summary_second_period.csv']

    analyzer = CSVAnalyzer(file_paths, output_filenames)
    analyzer.process_files()
