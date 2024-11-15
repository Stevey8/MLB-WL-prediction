import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pybaseball
from pybaseball import statcast
pybaseball.cache.enable()

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os


def get_away_batting_order(group):
    away_batters = group.loc[group['inning_topbot'] == 'Top', 'batter'].unique()[:9]
    away_b_dict = {f'away_b{i+1}':away_batters[i] for i in range(9)}
    return pd.Series(away_b_dict)

def get_home_batting_order(group):
    home_batters = group.loc[group['inning_topbot'] == 'Bot', 'batter'].unique()[:9]
    home_b_dict = {f'home_b{i+1}':home_batters[i] for i in range(9)}
    return pd.Series(home_b_dict)


def collect_pitch_level_data(start, end):
    df = statcast(start_dt= start,end_dt= end).reset_index().sort_index(ascending=False).reset_index(drop=True)
    return df

def collect_game_level_data(df):
    data_without_batters = df.groupby('game_pk',sort=False).apply(lambda group: pd.Series({
    'home_result': None,
    'date': group['game_date'].iloc[0],  # only one corresponding value
    'away_team': group['away_team'].iloc[0],  # only one corresponding value
    'home_team': group['home_team'].iloc[0],  # only one corresponding value
    'away_final_score': group['post_away_score'].iloc[-1],
    'home_final_score': group['post_home_score'].iloc[-1],
    'away_starting_pitcher': group.loc[(group['inning'] == 1) & (group['inning_topbot'] == 'Bot'), 'pitcher'].iloc[0],
    'home_starting_pitcher': group.loc[(group['inning'] == 1) & (group['inning_topbot'] == 'Top'), 'pitcher'].iloc[0],
    })).sort_values(by='date', ascending=True).reset_index()
    
    data_without_batters['home_result'] = np.where(data_without_batters['home_final_score'] > data_without_batters['away_final_score'], 'W', 'L')
    
    away_bs = df.groupby('game_pk',sort=False).apply(get_away_batting_order)
    home_bs = df.groupby('game_pk',sort=False).apply(get_home_batting_order)

    data_with_batters = data_without_batters.merge(away_bs, on='game_pk').merge(home_bs, on='game_pk')
    
    return data_with_batters


def get_batting_metrics(group):
    ab_events = ['single', 'double', 'triple', 'home_run', 'strikeout', 'field_out', 
                'grounded_into_double_play', 'double_play', 'triple_play', 'strikeout_double_play',
                'fielders_choice', 'fielders_choice_out', 'field_error', 'force_out']
    
    ab_count = group[group['events'].isin(ab_events)].shape[0]
    bb_count = group[group['events'] == 'walk'].shape[0]
    hbp_count = group[group['events'] == 'hit_by_pitch'].shape[0]
    single_count = group[group['events'] == 'single'].shape[0]
    double_count = group[group['events'] == 'double'].shape[0]
    triple_count = group[group['events'] == 'triple'].shape[0]
    home_run_count = group[group['events'] == 'home_run'].shape[0]
    sac_fly_count = group[group['events'].isin(['sac_fly', 'sac_fly_double_play'])].shape[0]

    return pd.Series({
        'ab': ab_count,
        'bb': bb_count,
        'hbp': hbp_count,
        'single': single_count,
        'double': double_count,
        'triple': triple_count,
        'hr': home_run_count,
        'sf': sac_fly_count
    })

def get_player_game_batting(df, filename = "", savefile = False):
    batting_away = df[df['inning_topbot']=='Top'].groupby(['game_pk','batter'],sort=False).apply(get_batting_metrics)
    batting_home = df[df['inning_topbot']=='Bot'].groupby(['game_pk','batter'],sort=False).apply(get_batting_metrics)
    
    batting_away_lineup = batting_away.groupby('game_pk').head(9)
    batting_home_lineup = batting_home.groupby('game_pk').head(9)

    if savefile:
        batting_away_lineup.to_csv(filename + "_away.csv", index=True)
        batting_home_lineup.to_csv(filename + "_home.csv", index=True)
        
    return batting_home_lineup, batting_away_lineup

def get_game_batter_stats(data, batting_stats, batter_stats = ["ab", "bb", "hbp", "single", "double", "triple", "hr", "sf"]):
    
    batters = ["away_b1", "away_b2", "away_b3", "away_b4", "away_b5", "away_b6", "away_b7", "away_b8", "away_b9", "home_b1", "home_b2", "home_b3", "home_b4", "home_b5", "home_b6", "home_b7", "home_b8", "home_b9"]
    
    # Append stats for every batter and clean
    for batter in batters:
        data = data.merge(batting_stats, how="left", left_on= ["game_pk",batter], right_on=["game_pk","batter"], suffixes=("","_"+batter))
    
    data = data.rename(columns = lambda col: col + "_away_b1" if col in batter_stats else col)

    for batter in batters:
        data = data.drop(columns=["Date_"+ batter, "mlbID_" + batter])
    
    data = data.drop(columns = ["home_result", "date", "away_team", "home_team", "away_score", "home_score"])

    return data

def get_previous_n_games(games, n, filename = "", save_file = False):
    results = pd.read_csv("data_with_batter_stats.csv")
    home_teams, away_teams = results.copy(), results.copy()


    home_teams["team"] = home_teams["home_team"]
    away_teams["team"] = away_teams["away_team"]

    games = pd.concat([home_teams[["game_pk","date","team"]], away_teams[["game_pk","date","team"]]])

    games["date"] = pd.to_datetime(games["date"])

    games.sort_values(by = ["team","date"], inplace = True, axis = 0, ascending = [True, True])
    prev_col = "game_pk"

    for i in range(n):
        column_name = f"prev_{i}_game_pk"
        games[column_name] = games.groupby('team')[prev_col].shift()
        prev_col = column_name
    
    if save_file:
        games.to_csv(filename, index = False)
    
    return games

def add_prev_game_stats(games, N, game_batter_results):
    for i in range(N):
        game_id_col = f"prev_{i}_game_pk"
        games = games.merge(game_batter_results, how="left", left_on= game_id_col , right_on= "game_pk", suffixes=("",f"_{i}"))
    return games


def main():
    N = 10
    df = collect_pitch_level_data(start= '2023-03-30',end= '2023-10-01')
    print(1)
    game_data = collect_game_level_data(df)
    batting_home_lineup, batting_away_lineup = get_player_game_batting(df, savefile= False)
    print(2)
    # Batter stats for every game
    batter_stats = pd.concat([batting_home_lineup, batting_away_lineup])
    print(3)
    # add batter stats for each game
    game_batter_results = get_game_batter_stats(game_data, batter_stats)
    print(4)
    # Find previous games
    games = get_previous_n_games(game_data, N, save_file= False)
    print(5)
    training_data = add_prev_game_stats(games, N, game_batter_results)
    print(6)
    training_data.to_csv("previous_games.csv", index = False)   #Add previous game stats
    return


if __name__ == "__main__":
    main()
