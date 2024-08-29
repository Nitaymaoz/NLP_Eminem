from ModelTrainer import ModelTrainer
from SongsAnalyze import SongsAnalyze, load_lyrics_from_file
import pandas as pd
from Helper import Helper


class Main:
    def __init__(self):
        self.album_analyzer = SongsAnalyze()
        self.model_trainer = ModelTrainer()

        # Initialize constants
        self.curse_words = list(Helper.CURSE_WORDS)
        self.slang_words = list(Helper.Slang)
        self.names = list(Helper.Names)
        self.file_names = Helper.FILE_NAMES
        self.period_filenames = ['song_analysis_first_period.csv', 'song_analysis_second_period.csv']

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')

    def analyze_and_save(self, lyrics_df, all_lyrics, filename):
        song_analysis_results = []

        for index, row in lyrics_df.iterrows():
            song_title = row['Song Title']
            lyrics = row['Lyrics']
            album_name = row['Album']

            print(f"Analyzing song: {song_title} in album: {album_name}")
            song_analysis = self.album_analyzer.analyze_song(lyrics, all_lyrics)
            song_analysis_results.append({
                "Album": album_name,
                "Song Title": song_title,
                **song_analysis
            })
            for key, value in song_analysis.items():
                print(f"  {key}: {value}")
            print("\n")

        # Save song analysis to the corresponding CSV file for the period
        self.save_to_csv(song_analysis_results, filename)

    def run(self):
        for i, file_name in enumerate(self.file_names):
            lyrics_df = Helper.load_lyrics_from_file(file_name)
            all_lyrics = lyrics_df['Lyrics'].tolist()

            # Train models
            #self.model_trainer.train_models(self.curse_words, self.slang_words, self.names, all_lyrics)

            # Analyze and save results
            self.analyze_and_save(lyrics_df, all_lyrics, self.period_filenames[i])


if __name__ == "__main__":
    main = Main()
    main.run()
