from ModelTrainer import ModelTrainer
from SongsAnalyze import SongsAnalyze, load_lyrics_from_file
import pandas as pd
from Helper import Helper

class Main:
    def __init__(self):
        self.album_analyzer = SongsAnalyze()
        self.model_trainer = ModelTrainer()

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')

    def run(self):
        # Known CSV files
        file_names = Helper.FILE_NAMES

        # Training data for the models
        curse_words = list(Helper.CURSE_WORDS)
        slang_words = list(Helper.Slang)
        names = list(Helper.Names)

        # Train models
        self.model_trainer.train_models(curse_words, slang_words, names)

        song_analysis_results = []

        for file_name in file_names:
            # Load lyrics from the file
            lyrics_df = load_lyrics_from_file(file_name)

            for index, row in lyrics_df.iterrows():
                song_title = row['Song Title']
                lyrics = row['Lyrics']
                album_name = row['Album']

                print(f"Analyzing song: {song_title} in album: {album_name}")
                song_analysis = self.album_analyzer.analyze_song(lyrics)
                song_analysis_results.append({
                    "Album": album_name,
                    "Song Title": song_title,
                    **song_analysis
                })
                for key, value in song_analysis.items():
                    print(f"  {key}: {value}")
                print("\n")

        # Save song analysis to CSV
        self.save_to_csv(song_analysis_results, 'song_analysis.csv')

if __name__ == "__main__":
    main = Main()
    main.run()
