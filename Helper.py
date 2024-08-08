import re
import pandas as pd


class Helper:
    # Constants
    FILE_NAMES = ['./DB/LyricsFirstPeriod.csv', './DB/LyricsSecondPeriod.csv']
    CURSE_WORDS = {'fuck', 'shit', 'bitch', 'ass', 'damn', 'bastard', 'hell', 'dick', 'piss', 'crap', 'prick', 'cock',
                   'pussy', 'slut', 'douche', 'cunt', 'motherfucker', 'fucking', 'ballsack', 'nuts', 'pussy', 'gay',
                   'fat', 'shot', 'gun', 'shoot', 'Glock', 'puke', 'faggot', 'bullets', 'murderin', 'roaches',
                   'midgets',
                   'retard', 'retarded', 'suck', 'vomit', 'suck', 'groin', 'alcohol', 'alcoholic', 'hit', 'influence',
                   'fat', 'motherfuckin', 'stealing', 'stealin', 'smoke', 'knives', 'killin', 'killing', 'screw', 'ass',
                   'kicked','drugs','drug'}
    Slang = {'murderin', 'servin', 'mcs', 'tryin', 'brainiac', 'relaxin', 'motherfuckin', 'rhymin', 'slimin',
             'starin', 'gamblin', 'goin', 'steppin', 'slammin', 'droppin', 'stealin', 'usin', 'nigga',
             'niggas', 'jackin', 'knives', 'steamin', 'slicin', 'killin', 'gangsta', 'cancelin',
             'plannin', 'makin', 'ownin'}
    Names = {'shady', 'eminem', '2pac', 'biggie', 'kim', 'hailie', 'dre', 'kim', 'kendrick', 'santa claus',
             'jimmy carter', 'Jerry Springer', 'Hilary Clinton', 'Nicole', 'Norman Bates', 'Susan', 'Dave', 'John',
             'Kelly', 'Ron', 'Sue', 'marshall', 'mike tyson', 'pete', 'ron goldman', 'Marty Schottenheimer',
             'Outsidaz', 'Jenny Craig', 'Slim Shady', 'slim', 'momma', 'mom', 'Stan', 'Ronnie', 'Bonnie',
             'Eric', 'Rémy Martin', 'Mathers', 'Proof', 'Ludacris', 'Marilyn Manson', 'Michael Jackson', 'Reggie',
             'Kim Mathers', 'Bobby', 'R. Kelly', 'Fetty Wap', 'Bizarre', 'Tara Reid', 'Richard Ramirez', 'JAY', 'Elvis',
             'Jack the ripper', 'Jay-Z', 'Nas', 'paul', 'Bin Laden', 'mama', 'Diddy', 'Billie Eilish','stevie','nate'}

    @staticmethod
    def preprocess_lyrics(lyrics):
        """
        Removes text inside brackets from the lyrics.
        """
        return re.sub(r'\[.*?]', '', lyrics).strip()

    @staticmethod
    def load_lyrics_from_file(file_path):
        """
        Loads lyrics from a single CSV file and preprocesses them to remove text inside brackets.
        Assumes each CSV file has columns 'Song Title', 'Lyrics', and 'Album'.
        """
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df['Lyrics'] = df['Lyrics'].apply(Helper.preprocess_lyrics)
        return df
