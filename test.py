import pandas as pd 

if __name__ == "__main__":
    params_dict = {}
    df=pd.read_csv('./results/song_analysis_second_period.csv')
    for index, row in df.iterrows():
        for param in df.columns:
            params_dict[param] = row[param]

    print( df['Song Title'].to_list())