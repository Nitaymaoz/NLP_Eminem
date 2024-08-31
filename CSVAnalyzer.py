import pandas as pd
from collections import Counter
from Helper import Helper
import numpy as np

class CSVAnalyzer:
    def __init__(self, file_paths, album_summary_filenames, period_summary_filenames):
        self.file_paths = file_paths
        self.album_summary_filenames = album_summary_filenames
        self.period_summary_filenames = period_summary_filenames

    def load_data(self, file_path):
        return pd.read_csv(file_path)

    def process_files(self):
        period_results = []

        for i, file_path in enumerate(self.file_paths):
            print(f"Processing file: {file_path}")
            df = self.load_data(file_path)

            # Group by album and process each album individually
            grouped = df.groupby('Album')
            all_album_results = []

            for album, album_df in grouped:
                print(f"Analyzing Album: {album}")
                processed_results = self.analyze_album(album, album_df)
                all_album_results.append(processed_results)

            # Convert the list of dictionaries to a DataFrame and save to CSV
            final_df = pd.DataFrame(all_album_results)
            self.save_to_csv(final_df, self.album_summary_filenames[i])

            # Add the processed album results to the period summary
            period_results.append(self.analyze_period(final_df))

            # Save the period-level summaries
        for i, period_result in enumerate(period_results):
            self.save_to_csv(pd.DataFrame([period_result]), self.period_summary_filenames[i])

    def analyze_album(self, album, album_df):
        results = {'Album': album}

        # Average and Median for specified columns
        columns_to_average = ['total_unique_words', 'total_words', 'total_names',
                              'total_slangs', 'total_curse_words']
        for col in columns_to_average:
            mean_value = album_df[col].mean()
            std_value = album_df[col].std()
            cv_value = (std_value / mean_value) * 100 if mean_value != 0 else 0

            results[f'avg_{col}'] = round(mean_value, 2)
            results[f'median_{col}'] = round(album_df[col].median(), 2)
            results[f'std_{col}'] = round(std_value, 2)
            results[f'cv_{col}'] = round(cv_value, 2)  # Adding CV to the results

        # Summing and Counting for specified columns
        columns_to_sum = ['predicted_curse_words', 'predicted_slang_words',
                          'predicted_names', 'word_frequency']

        for col in columns_to_sum:
            summed_col = Counter()
            for entry in album_df[col]:
                summed_col.update(eval(entry))  # Using eval to convert string to dict
            results[f'summed_{col}'] = dict(summed_col)

        # Found album and song name references
        results['found_albums_names_refs'] = self.find_references(album_df, 'found_albums_names_refs')
        results['found_songs_names_refs'] = self.find_references(album_df, 'found_songs_names_refs')

        # Sentiment analysis
        sentiment = self.calculate_average_sentiment(album_df)
        results.update(sentiment)

        # Named entities average
        named_entities_avg = self.calculate_average_named_entities(album_df)
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
            sentiment_counter[word] = round(sentiment_counter[word]/len(sentiment_col), 2)  # Averaging per word

        return {'average_sentiment': dict(sentiment_counter)}

    def analyze_period(self, df):
        results = {}

        # Average and Median for specified columns
        columns_to_average = ['avg_total_unique_words', 'avg_total_words', 'avg_total_names',
                              'avg_total_slangs', 'avg_total_curse_words']
        for col in columns_to_average:
            mean_value = df[col].mean()
            std_value = df[col].std()
            cv_value = (std_value / mean_value) * 100 if mean_value != 0 else 0

            results[f'avg_{col}'] = round(mean_value, 2)
            results[f'median_{col}'] = round(df[col].median(), 2)
            results[f'std_{col}'] = round(std_value, 2)
            results[f'cv_{col}'] = round(cv_value, 2)  # Adding CV to the results

        # Summing and Counting for specified columns
        columns_to_sum = ['summed_predicted_curse_words', 'summed_predicted_slang_words',
                          'summed_predicted_names', 'summed_word_frequency']

        for col in columns_to_sum:
            summed_col = Counter()
            for entry in df[col]:
                summed_col.update(entry)  # Already a dict, no need to use eval
            results[f'total_{col}'] = dict(summed_col)

        return results

    def calculate_average_named_entities(self, df):
        named_entities_col = df['named_entities']
        entity_counter = Counter()

        for entry in named_entities_col:
            entities_dict = eval(entry)
            entity_counter.update(entities_dict)

        for entity in entity_counter:
            entity_counter[entity] /= round(len(named_entities_col), 2)  # Averaging per entity, round to 2 points after decimal point

        return dict(entity_counter)

    def save_to_csv(self, df, filename):
        df.to_csv(filename, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    file_paths = Helper.analysis_file_paths
    album_summary_filenames = Helper.summary_filenames_by_album
    period_summary_filenames = Helper.summary_filenames_by_period

    analyzer = CSVAnalyzer(file_paths, album_summary_filenames,period_summary_filenames)
    analyzer.process_files()
