from SongsAnalyze import SongsAnalyze, load_lyrics_from_file
import pandas as pd


class Main:
    def __init__(self):
        self.album_analyzer = SongsAnalyze()

    def save_to_csv(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')

    def run(self):
        # Known CSV files
        file_names = ['./DB/LyricsFirstPeriod.csv', './DB/LyricsSecondPeriod.csv']

        song_analysis_results = []

        for file_name in file_names:
            # Load lyrics from the file
            lyrics_df = load_lyrics_from_file(file_name)

            # Group by album without including the grouping columns
            albums = lyrics_df.groupby('Album')[['Song Title', 'Lyrics']].apply(lambda df: df.values.tolist()).to_dict()

            for album_name, album_data in albums.items():
                print(f"Analyzing album: {album_name}")
                song_titles = [song[0] for song in album_data]
                album_lyrics = [song[1] for song in album_data]
                album_analysis = self.album_analyzer.analyze_album(album_lyrics, song_titles)
                for song_title, song_analysis in album_analysis:
                    print(f"Analysis for {song_title} in {album_name}:")
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
