import ast
import operator
import pandas as pd
import difflib
from SongsAnalyze import SongsAnalyze
from Helper import Helper


class ourAlgorithm:
    def __init__(self, song, firstPeriod='./results/first_period_summary.csv',
                 secondPeriod='./results/second_period_summary.csv'):

        self.first = pd.read_csv(firstPeriod)
        self.second = pd.read_csv(secondPeriod)
        self.firstFinalScore = 0
        self.secondFinalScore = 0
        self.weights = {
            'word_frequency': 10,
            'named_entities': 2,
            'dependency_parse': 8,
            'topic_modeling': 1,
            'non_real_words_freq': 4,
            'curse_words_freq': 15,
            'sentiment_analysis': 15,
            'predicted_curse_words': 15,
            'predicted_slang_words': 15,
            'predicted_names': 15,
            'total_unique_words': 10,
            'total_words': 10,
            'total_names': 19,
            'total_slangs': 10
        }
        songAnalyzer = SongsAnalyze()
        self.songAnalyze = songAnalyzer.analyze_song(song)
        self.allSongsSecond = Helper.songs_second_period

    # def hasToBeSecond(self, songsRefs):
    #     for song in songsRefs:
    #         if song in self.allSongsSecond:
    #             return True
    #     return False

    def whoWins(self, val1, val2, input):
        middle = (val1 + val2) / 2
        if input < middle:
            if val1 < val2:
                return [1, 0]
            else:
                return [0, 1]
        else:
            if val1 > val2:
                return [1, 0]
            else:
                return [0, 1]

    def recommend(self):
        # if self.hasToBeSecond(self.songAnalyze['found_songs_names_refs']):
        #     return "Second Period"

        for parameter, weight in self.weights.items():
            # print("still working")
            if parameter in ['total_curse_words', 'total_unique_words', 'total_words', 'total_names', 'total_slangs']:
                if parameter == 'total_curse_words':
                    result = self.whoWins(
                        self.first['avg_avg_total_curse_words'][0],
                        self.second['avg_avg_total_curse_words'][0],
                        self.songAnalyze[parameter]
                    )
                elif parameter == 'total_unique_words':
                    result = self.whoWins(
                        self.first['avg_total_unique_words'][0],
                        self.second['avg_total_unique_words'][0],
                        self.songAnalyze[parameter]
                    )
                elif parameter == 'total_words':
                    result = self.whoWins(
                        self.first['avg_total_words'][0],
                        self.second['avg_total_words'][0],
                        self.songAnalyze[parameter]
                    )
                elif parameter == 'total_names':
                    result = self.whoWins(
                        self.first['avg_total_names'][0],
                        self.second['avg_total_names'][0],
                        self.songAnalyze[parameter]
                    )
                elif parameter == 'total_slangs':
                    result = self.whoWins(
                        self.first['avg_total_slangs'][0],
                        self.second['avg_total_slangs'][0],
                        self.songAnalyze[parameter]
                    )

                self.firstFinalScore += result[0] * weight * (1 / self.first[f'cv_avg_{parameter}'][0])
                self.secondFinalScore += result[1] * weight * (1 / self.second[f'cv_avg_{parameter}'][0])

        if self.firstFinalScore < self.secondFinalScore:
            return "Second Period"
        return "First Period"


if __name__ == "__main__":
    firstPeriod = './DB/LyricsFirstPeriod.csv'
    firstDF = pd.read_csv(firstPeriod)
    count = 0
    allsongs = 0
    for index, row in firstDF.iterrows():
        allsongs += 1
        print(allsongs)
        alg = ourAlgorithm(row['Lyrics'])
        ans = alg.recommend()
        if ans == 'First Period':
            count += 1


    print(f"Score for first period: {count / allsongs * 100:.2f}%")

    secondPeriod = './DB/LyricsSecondPeriod.csv'
    secondDF = pd.read_csv(secondPeriod)
    count = 0
    allsongs = 0
    for index, row in secondDF.iterrows():
        allsongs += 1
        print(allsongs)
        alg = ourAlgorithm(row['Lyrics'])
        ans = alg.recommend()
        if ans == 'Second Period':
            count += 1


    print(f"Score for second period: {count / allsongs * 100:.2f}%")
