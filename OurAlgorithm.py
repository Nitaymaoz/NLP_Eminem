import ast
import operator
import pandas as pd 
import difflib
from SongsAnalyze import SongsAnalyze

class ourAlgorithm:
    def __init__(self , song, firstPeriod='./results/period_summary_first_period.csv' ,secondPeriod='./results/period_summary_second_period.csv' ):
            
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
                            'total_unique_words' : 10, 
                            'total_words' : 10,
                            'total_names' : 19,
                            'total_slangs' : 10
                        }
            songAnalyzer = SongsAnalyze()
            self.songAnalyze = songAnalyzer.analyze_song(song)
            self.allSongsSecond = self.second['Song Title'].to_list()
      
    def hasToBeSecond(self, songsRefs):
        for song in songsRefs:
                if song in self.allSongsSecond:
                      return True
        return False
      
    def whoWins (val1, val2, input):
        middle= (val1+val2)/2
        if (input<middle):
              if (val1<val2):
                   return [1,0]
              else:
                   return [1,0]
        else:
            if (val1>val2):
                   return [1,0]
            else:
                   return [1,0]
            
      
    def recommend(self):
            if self.hasToBeSecond(SongsAnalyze['found_songs_names_refs']):
                  return "Second Period"
                  
            for parameter , value in self.weights:
                if parameter == 'total_curse_words':
                      retult= self.whoWins(self.first['avg_avg_total_curse_words'][0] ,self.second['avg_avg_total_curse_words'][0] , SongsAnalyze[parameter] )
                      self.firstFinalScore= self.firstFinalScore +retult[0]*value*(1/self.first['cv_avg_total_slangs'][0])
                      self.secondFinalScore= self.secondFinalScore +retult[1]*value*(1/self.second['cv_avg_total_slangs'][0])

                if parameter == 'total_unique_words':
                      retult= self.whoWins(self.first['avg_total_unique_words'][0] ,self.second['avg_total_unique_words'][0] , SongsAnalyze[parameter] )
                      self.firstFinalScore= self.firstFinalScore +retult[0]*value*(1/self.first['cv_avg_total_unique_words'][0])
                      self.secondFinalScore= self.secondFinalScore +retult[1]*value*(1/self.second['cv_avg_total_unique_words'][0])
                
                if parameter == 'total_words':
                      retult= self.whoWins(self.first['avg_total_words'][0] ,self.second['avg_total_words'][0] , SongsAnalyze[parameter] )
                      self.firstFinalScore= self.firstFinalScore +retult[0]*value*(1/self.first['cv_avg_total_words'][0])
                      self.secondFinalScore= self.secondFinalScore +retult[1]*value*(1/self.second['cv_avg_total_words'][0])

                if parameter == 'total_names':
                      retult= self.whoWins(self.first['avg_total_names'][0] ,self.second['avg_total_names'][0] , SongsAnalyze[parameter] )
                      self.firstFinalScore= self.firstFinalScore +retult[0]*value*(1/self.first['cv_avg_total_names'][0])
                      self.secondFinalScore= self.secondFinalScore +retult[1]*value*(1/self.second['cv_avg_total_names'][0])

                if parameter == 'total_slangs':
                      retult= self.whoWins(self.first['avg_total_slangs'][0] ,self.second['avg_total_slangs'][0] , SongsAnalyze[parameter] )
                      self.firstFinalScore= self.firstFinalScore +retult[0]*value*(1/self.first['cv_avg_total_slangs'][0])
                      self.secondFinalScore= self.secondFinalScore +retult[1]*value*(1/self.second['cv_avg_total_slangs'][0])


            if self.firstFinalScore < self.secondFinalScore:
                 return "Second Period"
            return "First Period"
    
    

      