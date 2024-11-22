import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pybaseball
from pybaseball import statcast
pybaseball.cache.enable()

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

def get_pitch_level_data(start, end):
    df = statcast(start_dt=start,end_dt=end).iloc[::-1].reset_index(drop=True)
    return df

def get_game_level_data(df):
    data_without_batters = df.groupby('game_pk').apply(lambda group: pd.Series({
    'home_result': None,
    'date': group['game_date'].iloc[0],  # only one corresponding value
    'away_team': group['away_team'].iloc[0],  # only one corresponding value
    'home_team': group['home_team'].iloc[0],  # only one corresponding value
    'away_score': group['away_score'].max(),
    'home_score': group['home_score'].max(),
    'away_starting_pitcher': group.loc[(group['inning'] == 1) & (group['inning_topbot'] == 'Bot'), 'pitcher'].iloc[0],
    'home_starting_pitcher': group.loc[(group['inning'] == 1) & (group['inning_topbot'] == 'Top'), 'pitcher'].iloc[0],
    })).sort_values(by='date', ascending=True).reset_index()
    data_without_batters['home_result'] = np.where(data_without_batters['home_score'] > data_without_batters['away_score'], 'W', 'L')
    return data_without_batters

def get_game_level_data_with_batters(df, data_without_batters):
    def get_away_batting_order(group):
        away_batters = group.loc[group['inning_topbot'] == 'Top', 'batter'].unique()[:9]
        away_b_dict = {f'away_b{i+1}':away_batters[i] for i in range(9)}
        return pd.Series(away_b_dict)

    def get_home_batting_order(group):
        home_batters = group.loc[group['inning_topbot'] == 'Bot', 'batter'].unique()[:9]
        home_b_dict = {f'home_b{i+1}':home_batters[i] for i in range(9)}
        return pd.Series(home_b_dict)


    away_bs = df.groupby('game_pk').apply(get_away_batting_order)
    home_bs = df.groupby('game_pk').apply(get_home_batting_order)
    data_with_batters = data_without_batters.merge(away_bs, on='game_pk').merge(home_bs, on='game_pk')
    return data_with_batters

def get_previous_n_batter_games(results, n, filename = "", save_file = False):
    home_teams, away_teams = results.copy(), results.copy()


    home_teams["team"] = home_teams["home_team"]
    away_teams["team"] = away_teams["away_team"]
    games = pd.concat([home_teams, away_teams])

    games = games.loc[:,['team', 'date', 'game_pk']]

    games["date"] = pd.to_datetime(games["date"])

    games.sort_values(by = ["team","date"], inplace = True, axis = 0, ascending = [True, True])
    prev_col = "game_pk"

    for i in range(n):
        column_name = f"prev_{i+1}_game_pk"
        games[column_name] = games.groupby('team')[prev_col].shift()
        prev_col = column_name
    
    if save_file:
        games.to_csv(filename, index = False)
    
    return games

def get_previous_n_pitcher_games(results, n, filename = "", save_file = False):
    home_teams, away_teams = results.copy(), results.copy()


    home_teams["team"] = home_teams["home_team"]
    away_teams["team"] = away_teams["away_team"]

    home_teams["pitcher"] = home_teams["home_starting_pitcher"]
    away_teams["pitcher"] = away_teams["away_starting_pitcher"]


    games = pd.concat([home_teams, away_teams])

    games = games.loc[:,['team', 'date', 'game_pk', 'pitcher']]

    games["date"] = pd.to_datetime(games["date"])

    games.sort_values(by = ["team","date", "pitcher"], inplace = True, axis = 0, ascending = [True, True, True])
    prev_col = "game_pk"

    for i in range(n):
        column_name = f"prev_{i+1}_game_pk"
        games[column_name] = games.groupby(['team', 'pitcher'])[prev_col].shift()
        prev_col = column_name
    
    if save_file:
        games.to_csv(filename, index = False)
    
    return games


def add_prev_game_stats(games, batter_stats, pitcher_stats, prev_batter_games, prev_pitcher_games, N):
    # Reshape batter stats
    batter_stats.drop(columns=['Unnamed: 0', 'game_sequence','batter'], inplace=True)
    df_transformed = batter_stats.melt(id_vars=['game_pk', 'team','batting_order'], var_name='stat', value_name='value')

    df_pivot = df_transformed.pivot_table(
        index= ['game_pk', 'team'], 
        columns=['stat', 'batting_order'], 
        values='value'
    )

    # Flatten the MultiIndex columns to a single level
    df_pivot.columns = [f"{stat}_batter{batter}" for stat, batter in df_pivot.columns]
    batter_stats = df_pivot.reset_index()
    
    games = games.merge(prev_batter_games, how="left", left_on= ["game_pk", "home_team"], right_on= ["game_pk", "team"], suffixes=("","_home_batter"))
    games = games.merge(prev_batter_games, how="left", left_on= ["game_pk", "away_team"], right_on= ["game_pk", "team"], suffixes=("","_away_batter"))
    
    games = games.merge(prev_pitcher_games, how="left", left_on= ["game_pk", "home_team"], right_on= ["game_pk", "team"], suffixes=("","_home_pitcher"))
    games = games.merge(prev_pitcher_games, how="left", left_on= ["game_pk", "away_team"], right_on= ["game_pk", "team"], suffixes=("","_away_pitcher"))

    games.drop(columns=['team', 'date_home_batter','team_away_batter', 'date_away_batter','team_home_pitcher', 'date_home_pitcher', 'pitcher','team_away_pitcher', 'date_away_pitcher', 'pitcher_away_pitcher','away_starting_pitcher', 'home_starting_pitcher'], inplace=True)
    
    games.rename(columns={
        "prev_1_game_pk":"prev_1_game_pk_home_batter",
        "prev_2_game_pk":"prev_2_game_pk_home_batter",
        "prev_3_game_pk":"prev_3_game_pk_home_batter",
        "prev_4_game_pk":"prev_4_game_pk_home_batter",
        "prev_5_game_pk":"prev_5_game_pk_home_batter",
        "prev_6_game_pk":"prev_6_game_pk_home_batter",
        "prev_7_game_pk":"prev_7_game_pk_home_batter",
        "prev_8_game_pk":"prev_8_game_pk_home_batter",
        "prev_9_game_pk":"prev_9_game_pk_home_batter",
        "prev_10_game_pk":"prev_10_game_pk_home_batter"
    }, inplace=True)

    for i in range(N):
        home_team_col = f"prev_{i+1}_game_pk_home_batter"
        away_team_col = f"prev_{i+1}_game_pk_away_batter"

        ## Add home_team batter stats
        games = games.merge(batter_stats, how="left", left_on= [home_team_col , "home_team"], right_on= ["game_pk", "team"], suffixes=("",f"_home_{i+1}"))
        
        games = games.merge(batter_stats, how="left", left_on= [away_team_col , "away_team"], right_on= ["game_pk", "team"], suffixes=("",f"_away_{i+1}"))

        home_team_col = f"prev_{i+1}_game_pk_home_pitcher"
        away_team_col = f"prev_{i+1}_game_pk_away_pitcher"

        home_key_cols = [home_team_col, "home_team"]
        away_key_cols = [away_team_col, "away_team"]

               
        # Add home_team pitcher stats
        games = games.merge(pitcher_stats[["game_pk", "team","is_host", "n_pitches", "ip", "er", "k", "bb", "h"]], how="left", left_on= home_key_cols, right_on= ["game_pk", "team"], suffixes=("",f"_home_pitcher_{i+1}"))
        # Add away_team pitcher stats
        games = games.merge(pitcher_stats[["game_pk", "team","is_host", "n_pitches", "ip", "er", "k", "bb", "h"]], how="left", left_on= away_key_cols, right_on= ["game_pk", "team"], suffixes=("",f"_away_pitcher_{i+1}"))
    
    #games.drop(columns=["Unamed: 0"], inplace=True)
    
    return games



def main():
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
    
    for year in years:
        print(f"Processing year {year}")
        game_batter_data = pd.read_csv(f"data/df_{year}_game_info.csv")

        prev_batter_games = get_previous_n_batter_games(game_batter_data, 10, save_file = False)
        prev_pitcher_games = get_previous_n_pitcher_games(game_batter_data, 10, save_file = False)

        batter_data = pd.read_csv(f"data/df_{year}_batting_stats.csv")
        pitcher_data = pd.read_csv(f"data/df_{year}_pitching_stats.csv")
        pitcher_data.rename(columns = {'is_home': 'is_host'}, inplace = True)


        games_with_hist_stats = add_prev_game_stats(game_batter_data, batter_data, pitcher_data, prev_batter_games, prev_pitcher_games, 10)
        print(f"Saving data for year {year}")
        games_with_hist_stats.to_csv(f"data/games_with_hist_stats_{year}.csv", index = False)

if __name__ == "__main__":
    main()

