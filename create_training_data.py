import pandas as pd
import numpy as np
import argparse

def clean_unnecessary_cols(orig_data):
    data = orig_data.copy()
    drop_cols = ['Unnamed: 0', 'date','home_team', 'away_team','game_pk', 'away_final_score', 'home_final_score', 'away_b1', 'away_b2', 'away_b3', 'away_b4', 'away_b5', 'away_b6', 'away_b7', 'away_b8', 'away_b9', 'home_b1', 'home_b2', 'home_b3', 'home_b4', 'home_b5', 'home_b6', 'home_b7', 'home_b8', 'home_b9', 'prev_1_game_pk_home_batter', 'prev_2_game_pk_home_batter', 'prev_3_game_pk_home_batter', 'prev_4_game_pk_home_batter', 'prev_5_game_pk_home_batter', 'prev_6_game_pk_home_batter', 'prev_7_game_pk_home_batter', 'prev_8_game_pk_home_batter', 'prev_9_game_pk_home_batter', 'prev_10_game_pk_home_batter', 'prev_1_game_pk_away_batter', 'prev_2_game_pk_away_batter', 'prev_3_game_pk_away_batter', 'prev_4_game_pk_away_batter', 'prev_5_game_pk_away_batter', 'prev_6_game_pk_away_batter', 'prev_7_game_pk_away_batter', 'prev_8_game_pk_away_batter', 'prev_9_game_pk_away_batter', 'prev_10_game_pk_away_batter', 'prev_1_game_pk_home_pitcher', 'prev_2_game_pk_home_pitcher', 'prev_3_game_pk_home_pitcher', 'prev_4_game_pk_home_pitcher', 'prev_5_game_pk_home_pitcher', 'prev_6_game_pk_home_pitcher', 'prev_7_game_pk_home_pitcher', 'prev_8_game_pk_home_pitcher', 'prev_9_game_pk_home_pitcher', 'prev_10_game_pk_home_pitcher', 'prev_1_game_pk_away_pitcher', 'prev_2_game_pk_away_pitcher', 'prev_3_game_pk_away_pitcher', 'prev_4_game_pk_away_pitcher', 'prev_5_game_pk_away_pitcher', 'prev_6_game_pk_away_pitcher', 'prev_7_game_pk_away_pitcher', 'prev_8_game_pk_away_pitcher', 'prev_9_game_pk_away_pitcher', 'prev_10_game_pk_away_pitcher', 'game_pk_home_1', 'team','is_home_batter1','is_home_batter2', 'is_home_batter3', 'is_home_batter4', 'is_home_batter5', 'is_home_batter6', 'is_home_batter7', 'is_home_batter8', 'is_home_batter9']
    data.drop(columns=drop_cols, inplace=True)

    for i in range(1,11):
        # Clean batter cols
        drop_batter_cols = [f'game_pk_away_{i}', f'team_away_{i}', f'is_home_batter1_away_{i}', f'is_home_batter2_away_{i}', f'is_home_batter3_away_{i}', f'is_home_batter4_away_{i}', f'is_home_batter5_away_{i}', f'is_home_batter6_away_{i}', f'is_home_batter7_away_{i}', f'is_home_batter8_away_{i}', f'is_home_batter9_away_{i}']
        data.drop(columns=drop_batter_cols, inplace=True)
        drop_batter_cols = [f'game_pk_home_{i}', f'team_home_{i}', f'is_home_batter1_home_{i}', f'is_home_batter2_home_{i}', f'is_home_batter3_home_{i}', f'is_home_batter4_home_{i}', f'is_home_batter5_home_{i}', f'is_home_batter6_home_{i}', f'is_home_batter7_home_{i}', f'is_home_batter8_home_{i}', f'is_home_batter9_home_{i}']
        try:
            data.drop(columns=drop_batter_cols, inplace=True)
        except:
            print(f"No columns to drop for batter {i}")

        # Clean pitcher cols
        drop_pitcher_cols = [f'game_pk_home_pitcher_{i}', f'team_home_pitcher_{i}',f'game_pk_away_pitcher_{i}', f'team_away_pitcher_{i}']
        data.drop(columns=drop_pitcher_cols, inplace=True)

    data["home_outcome"] = data["home_result"].apply(lambda x: 1 if x == "W" else 0)
    data.drop(columns=["home_result"], inplace=True)

    suffix = "_home_1"
    data.columns = [
    col + suffix if i < 72 else col
    for i, col in enumerate(data.columns)
]
    suffix = "_home_pitcher_1"
    data.columns = [
    col + suffix if i > 143 and i < 151 else col
    for i, col in enumerate(data.columns)
]
    return data


def drop_pitcher_games(data, n, N):
    new_df = data.copy()
    for i in range(n+1,N+1):
        new_df.drop(new_df.filter(regex=f'pitcher_{i}$').columns, axis=1, inplace = True)
    return new_df


def drop_batter_games(data, n, N):
    new_df = data.copy()
    for i in range(n+1,N+1):
        new_df.drop(new_df.filter(regex=f"batter[1-9]_(home|away)_{i}$").columns, axis=1, inplace = True)
    return new_df

def data_augmentation(data):
    new_df = data.copy()
    new_df.rename(columns = lambda x: x.replace("home", "temp"), inplace = True)
    new_df.rename(columns = lambda x: x.replace("away", "home"), inplace = True)
    new_df.rename(columns = lambda x: x.replace("temp", "away"), inplace = True)
    
    new_df["home_outcome"] = new_df["away_outcome"].apply(lambda x: 0 if x == 1 else 1)
    new_df.drop(columns=["away_outcome"], inplace=True)
    
    data_reset = data.reset_index(drop=True)
    new_df_reset = new_df.reset_index(drop=True)

    #print(list(new_df.columns))
    #print(list(data.columns))

    augmented_df = pd.concat([data_reset, new_df_reset], ignore_index=True)
    return augmented_df

def main():
    # Collect arguments
    parser = argparse.ArgumentParser(description="Process some arguments.")

    # Add arguments
    parser.add_argument("b", type=int, help="Number of batter games to keep")
    parser.add_argument("p", type=int, help="The second argument")

    # Parse the arguments
    args = parser.parse_args()

    data_dir = "data/"
    years = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]

    data = []
    for year in years:
        year_data = pd.read_csv(f"{data_dir}games_with_hist_stats_{year}.csv")
        data.append(year_data)
        print(year_data.shape)
    

    data = pd.concat(data, ignore_index=True)

    cleaned_data = clean_unnecessary_cols(data)
    cleaner_data = drop_pitcher_games(cleaned_data, args.p, 10).dropna()
    cleaner_data = drop_batter_games(cleaner_data, args.b, 10).dropna()

    full_data = data_augmentation(cleaner_data)
    full_data.to_csv("data/full_training_data.csv", index = False)

if __name__ == "__main__":
    main()


