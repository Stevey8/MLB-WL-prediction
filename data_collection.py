import warnings
import pybaseball
from pybaseball import statcast
pybaseball.cache.enable()
import pandas as pd 
import numpy as np 


warnings.simplefilter(action='ignore', category=FutureWarning)

# Get pitch level data
def collect_pitch_level_data(start, end):
    """
    Collects pitch-by-pitch MLB data for a specified date range using pybaseball's statcast function.

    Args:
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        pandas.DataFrame: DataFrame containing detailed pitch-level statistics, sorted in reverse chronological order
    """
    df = statcast(start_dt= start,end_dt= end).reset_index().sort_index(ascending=False).reset_index(drop=True)
    return df

# Convert pitch level to game level
def get_away_batting_order(group):
    """
    Gets the batting order for the away team from a group of pitch-by-pitch data.

    Args:
        group (pandas.DataFrame): A DataFrame containing pitch-by-pitch data for a single game

    Returns:
        pandas.Series: A Series containing the first 9 unique batters that appeared for the away team,
                      with keys 'away_b1' through 'away_b9'
    """
    away_batters = group.loc[group['inning_topbot'] == 'Top', 'batter'].unique()[:9]
    away_b_dict = {f'away_b{i+1}':away_batters[i] for i in range(9)}
    return pd.Series(away_b_dict)


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

def get_home_batting_order(group):
    """
    Gets the batting order for the home team from a group of pitch-by-pitch data.
    """
    home_batters = group.loc[group['inning_topbot'] == 'Bot', 'batter'].unique()[:9]
    home_b_dict = {f'home_b{i+1}':home_batters[i] for i in range(9)}
    return pd.Series(home_b_dict)

def group_by_game(df):
    """
    Groups pitch-by-pitch data by game and extracts relevant game-level statistics.
    """
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
    
    data_without_batters['home_result'] = np.where(data_without_batters['home_final_score'] > data_without_batters['away_final_score'], 'W', 'L')

    away_bs = df.groupby('game_pk',sort=False).apply(get_away_batting_order)
    home_bs = df.groupby('game_pk',sort=False).apply(get_home_batting_order)

    data_with_batters = data_without_batters.merge(away_bs, on='game_pk').merge(home_bs, on='game_pk')

    return data_with_batters

def get_player_game_batting(df, filename = "", savefile = False):
    """
    Groups pitch-by-pitch data by game and extracts relevant batting statistics for each batter.
    """
    batting_away = df[df['inning_topbot']=='Top'].groupby(['game_pk','batter'],sort=False).apply(get_batting_metrics)
    batting_home = df[df['inning_topbot']=='Bot'].groupby(['game_pk','batter'],sort=False).apply(get_batting_metrics)
    
    batting_away_lineup = batting_away.groupby('game_pk').head(9)
    batting_home_lineup = batting_home.groupby('game_pk').head(9)

    if savefile:
        batting_away_lineup.to_csv(filename + "_away.csv", index=True)
        batting_home_lineup.to_csv(filename + "_home.csv", index=True)

    return batting_home_lineup, batting_away_lineup

def add_bat_stats_to_games(data, batting_stats, batter_stats = None):
    """
    Adds batting statistics to game-level data for each batter.
    """

    if batter_stats is None:
        batter_stats = ["ab", "bb", "hbp", "single", "double", "triple", "hr", "sf"]

    batters = ["away_b1", "away_b2", "away_b3", "away_b4", "away_b5", "away_b6", "away_b7", "away_b8", "away_b9", "home_b1", "home_b2", "home_b3", "home_b4", "home_b5", "home_b6", "home_b7", "home_b8", "home_b9"]
    batter_stats.append("batter")

    # Append stats for every batter and clean
    for batter in batters:
        data = data.merge(batting_stats, how="left", left_on= ["game_pk",batter], right_on=["game_pk","batter"], suffixes=("","_"+batter))
    
    data = data.rename(columns = lambda col: col + "_away_b1" if col in batter_stats else col)

    for batter in batters:
        data = data.drop(columns=["batter_" + batter])

    home_stats, away_stats = data.copy(), data.copy()
    home_stats = home_stats.loc[:, ~home_stats.columns.str.contains("away")]
    away_stats = away_stats.loc[:, ~away_stats.columns.str.contains("home")]

    home_stats.drop(columns = ['home_result', 'home_final_score'], inplace = True)
    away_stats.drop(columns = ['away_final_score'], inplace = True)

    for i in range(9):
        home_stats.drop(columns=f"home_b{i+1}", inplace=True)
        away_stats.drop(columns=f"away_b{i+1}", inplace=True)

    home_stats = home_stats.rename(
    columns={col: col.replace("home", "") for col in home_stats.columns if "home" in col}
)
    away_stats = away_stats.rename(
    columns={col: col.replace("away", "") for col in away_stats.columns if "away" in col}
)
    
    team_game_stats = pd.concat([home_stats, away_stats])
    team_game_stats.rename(columns = {"_team": "team"}, inplace = True)
    
    return team_game_stats

def get_previous_n_games(results, n, filename = "", save_file = False):
    """
    Gets the previous n games for each team.
    """
    games = results[["game_pk","date","team"]]

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

def add_prev_game_stats(games, prev_games, N, batter_games_stats):
    """
    Adds previous game statistics to game-level data.
    """
    games = games.merge(prev_games, how="left", left_on=["game_pk","home_team"], right_on=["game_pk","team"], suffixes=("", "_home"))
    games = games.merge(prev_games, how="left", left_on=["game_pk","away_team"], right_on=["game_pk","team"], suffixes=("", "_away"))
    for i in range(N):
        home_team_col = f"prev_{i+1}_game_pk"
        away_team_col = f"prev_{i+1}_game_pk_away"
        # Add home_team stats
        games = games.merge(batter_games_stats, how="left", left_on= [home_team_col , "home_team"], right_on= ["game_pk", "team"], suffixes=("",f"_{i+1}"))
        # Add away_team stats
        games = games.merge(batter_games_stats, how="left", left_on= [away_team_col , "away_team"], right_on= ["game_pk", "team"], suffixes=("",f"_away_{i+1}"))
    games = games.drop(columns = ["team","away_b1", "away_b2", "away_b3", "away_b4", "away_b5", "away_b6", "away_b7", "away_b8", "away_b9", "home_b1", "home_b2", "home_b3", "home_b4", "home_b5", "home_b6", "home_b7", "home_b8", "home_b9", "date_home", "date_away","prev_1_game_pk", "prev_2_game_pk", "prev_3_game_pk", "prev_4_game_pk", "prev_5_game_pk", "prev_6_game_pk", "prev_7_game_pk", "prev_8_game_pk", "prev_9_game_pk", "prev_10_game_pk", "date_away", "team_away", "prev_1_game_pk_away", "prev_2_game_pk_away", "prev_3_game_pk_away", "prev_4_game_pk_away", "prev_5_game_pk_away", "prev_6_game_pk_away", "prev_7_game_pk_away", "prev_8_game_pk_away", "prev_9_game_pk_away", "prev_10_game_pk_away"])
        
    for i in range(N):
        games = games.drop(columns = [f"date_{i+1}", f"team_{i+1}",f"game_pk_{i+1}",f"date_away_{i+1}", f"team_away_{i+1}",f"game_pk_away_{i+1}"])
    return games



def main():
    """
    Main function to execute the data collection pipeline.
    """
    N = 10
    YEAR = 2022
    START_DATE = f'{YEAR}-04-07'
    END_DATE = f'{YEAR}-10-05'
    # Get pitch level data
    df = collect_pitch_level_data(start= START_DATE,end= END_DATE)

    # Group into games
    games = group_by_game(df)

    # Get batter stats per game
    print("Getting batter stats per game...")
    _,_ = get_player_game_batting(df, savefile = True, filename = f"batting_data_{YEAR}")
    #batter_stats = pd.concat([batting_home_lineup, batting_away_lineup])

    batting_home_lineup, batting_away_lineup = pd.read_csv(f"batting_data_{YEAR}_home.csv"), pd.read_csv(f"batting_data_{YEAR}_away.csv")
    batter_stats = pd.concat([batting_home_lineup, batting_away_lineup])
    
    games_with_batting_stats = add_bat_stats_to_games(games, batter_stats)
    prev_games = get_previous_n_games(games_with_batting_stats, N)
    games_with_prev_stats = add_prev_game_stats(games, prev_games, N, games_with_batting_stats)

    games_with_prev_stats.to_csv(f"games_with_prev_stats_{YEAR}.csv", index = False)

if __name__ == "__main__":
    main()