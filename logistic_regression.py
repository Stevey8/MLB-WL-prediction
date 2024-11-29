import warnings
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, message="DataFrame is highly fragmented")
warnings.filterwarnings("ignore", category=UserWarning, message="A column-vector y was passed when a 1d array was expected")
warnings.filterwarnings("ignore", category=PerformanceWarning, message="DataFrame is highly fragmented")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

import pickle

import math

# train test split and cross validation
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

def keep_n_prev_games(df, n_games):
    columns_to_keep = [
        col for col in df.columns
        if col.split('_')[-1].isdigit() and int(col.split('_')[-1]) <= n_games
    ]
    
    columns_to_keep += [col for col in df.columns if not col.split('_')[-1].isdigit()]

    return df[columns_to_keep]

def calc_sum_stats(df, weights):
    '''
    given a df and specifying n_games, calculate the sum of each stat for each player (pitcher/batter1~9)
    and add as new cols of df
    colname style: f'{stat}_{pos}_{team}_{nth_game}'
    '''
    batter_stats = ['ab','bb','hbp','single','double','triple','hr','sf']
    pitcher_stats = ['n-pitches','ip','er','k','bb','h']

    df_new = df.copy()

    for team in ['away','home']:
        for pos in ['batter','pitcher']:
            if pos == 'batter':
                for order in range(1,10):
                    for stat in batter_stats:
                        prefix = f"{stat}_{pos}{order}_{team}_"
                        matching_cols = [col for col in df.columns if col.startswith(prefix)]
                        total_col_name = f"{stat}_{pos}{order}_{team}_total"
                        df_new.loc[:, total_col_name] = (df[matching_cols]*weights).sum(axis=1)

            else: 
                for stat in pitcher_stats:
                    prefix = f"{stat}_{pos}_{team}_"
                    matching_cols = [col for col in df.columns if col.startswith(prefix)]
                    total_col_name = f"{stat}_{pos}_{team}_total"
                    df_new.loc[:, total_col_name] = (df[matching_cols]*weights).sum(axis=1)

    # drop all unneeded cols (that ends with '_n' where n is n previous games)
    columns_to_drop = [
        col for col in df_new.columns
        if col.split('_')[-1].isdigit()
    ]
    df_dropped = df_new.drop(columns=columns_to_drop)
   
    return df_dropped

# batting 
def calc_ops(ab,bb,hbp,single,double,triple,hr,sf):
    '''
    calculate on base plus slugging average of a single batter  
    to calculate the rolling average, 
    simply take in the TOTAL VALUE of each param (up to last n games)
    '''
    inputs = [ab, bb, hbp, single, double, triple, hr, sf]
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in inputs):
        return np.nan 
     
    h = single+double+triple+hr

    if ab + bb + hbp + sf == 0:
        obp = 0  # If denominator is 0, set OBP to 0
    else:
        obp = (h + bb + hbp) / (ab + bb + hbp + sf)
    
    if ab == 0:
        slg = 0  # If denominator is 0, set SLG to 0
    else:
        slg = (single + 2 * double + 3 * triple + 4 * hr) / ab

    return obp + slg

def calc_ba(ab,bb,hbp,single,double,triple,hr,sf):
    inputs = [ab, bb, hbp, single, double, triple, hr, sf]
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in inputs):
        return np.nan 
    
    h = single+double+triple+hr
    if ab == 0: 
        return 0
    return h/ab

def calc_obp(ab,bb,hbp,single,double,triple,hr,sf):
    inputs = [ab, bb, hbp, single, double, triple, hr, sf]
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in inputs):
        return np.nan 
    
    h = single+double+triple+hr
    if ab + bb + hbp + sf == 0:
        obp = 0  # If denominator is 0, set OBP to 0
    else:
        obp = (h + bb + hbp) / (ab + bb + hbp + sf)
    return obp

def calc_slg(ab,bb,hbp,single,double,triple,hr,sf):
    inputs = [ab, bb, hbp, single, double, triple, hr, sf]
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in inputs):
        return np.nan 
    
    if ab == 0:
        slg = 0  # If denominator is 0, set SLG to 0
    else:
        slg = (single + 2 * double + 3 * triple + 4 * hr) / ab
    return slg

# pitching

def calc_era(er,ip):
    '''
    calculate earned run average of a single pitcher
    to calculate the rolling average, 
    simply take in the TOTAL VALUE of each param (up to last 10 games)
    '''
    inputs = [er, ip]
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in inputs):
        return np.nan 
    
    if ip == 0: 
        ip = 0.33
    return (er/ip)*9

def calc_k9(k,ip):
    inputs = [k, ip]
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in inputs):
        return np.nan 
    
    if ip == 0: 
        ip = 0.33
    return (k/ip)*9

# def calc_bb9(bb,ip):
#     inputs = [bb, ip]
#     if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in inputs):
#         return np.nan 
    
#     if ip == 0: 
#         ip = 0.33
#     return (bb/ip)*9

def calc_whip(bb,h,ip):
    inputs = [bb, h, ip]
    if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in inputs):
        return np.nan 
    
    if ip == 0: 
        ip = 0.33
    return (bb+h)/ip

def make_metrics(df):

    home_outcome = df['home_outcome']
    df = df.copy() 

    suffixes = {
        int(col.split('_')[-1])
        for col in df.columns
        if col.split('_')[-1].isdigit()
    }
    n_games = len(suffixes)

    for team in ['away','home']:
        for game in range(1,max(1,n_games)+1):
            for pos in ['batter','pitcher']:
                if pos == 'batter':
                    for order in range(1,10):
                        if n_games == 0:
                            game = 'total'
                        df[f'batavg_{pos}{order}_{team}_{game}'] = df.apply(
                            lambda row: calc_ba(
                                row[f'ab_{pos}{order}_{team}_{game}'],
                                row[f'bb_{pos}{order}_{team}_{game}'],
                                row[f'hbp_{pos}{order}_{team}_{game}'],
                                row[f'single_{pos}{order}_{team}_{game}'],
                                row[f'double_{pos}{order}_{team}_{game}'],
                                row[f'triple_{pos}{order}_{team}_{game}'],
                                row[f'hr_{pos}{order}_{team}_{game}'],
                                row[f'sf_{pos}{order}_{team}_{game}']
                            ), axis = 1)
                        
                        df[f'obp_{pos}{order}_{team}_{game}'] = df.apply(
                            lambda row: calc_obp(
                                row[f'ab_{pos}{order}_{team}_{game}'],
                                row[f'bb_{pos}{order}_{team}_{game}'],
                                row[f'hbp_{pos}{order}_{team}_{game}'],
                                row[f'single_{pos}{order}_{team}_{game}'],
                                row[f'double_{pos}{order}_{team}_{game}'],
                                row[f'triple_{pos}{order}_{team}_{game}'],
                                row[f'hr_{pos}{order}_{team}_{game}'],
                                row[f'sf_{pos}{order}_{team}_{game}']
                            ), axis = 1)
                        
                        df[f'slg_{pos}{order}_{team}_{game}'] = df.apply(
                            lambda row: calc_slg(
                                row[f'ab_{pos}{order}_{team}_{game}'],
                                row[f'bb_{pos}{order}_{team}_{game}'],
                                row[f'hbp_{pos}{order}_{team}_{game}'],
                                row[f'single_{pos}{order}_{team}_{game}'],
                                row[f'double_{pos}{order}_{team}_{game}'],
                                row[f'triple_{pos}{order}_{team}_{game}'],
                                row[f'hr_{pos}{order}_{team}_{game}'],
                                row[f'sf_{pos}{order}_{team}_{game}']
                            ), axis = 1)

                else: 
                     if n_games == 0:
                            game = 'total'
                     df[f'era_{pos}_{team}_{game}'] = df.apply(
                         lambda row: calc_era(
                             row[f'er_{pos}_{team}_{game}'],
                             row[f'ip_{pos}_{team}_{game}'],
                         ), axis = 1)
                     
                     df[f'k9_{pos}_{team}_{game}'] = df.apply(
                         lambda row: calc_k9(
                             row[f'er_{pos}_{team}_{game}'],
                             row[f'k_{pos}_{team}_{game}'],
                         ), axis = 1)
                     
                     df[f'whip_{pos}_{team}_{game}'] = df.apply(
                         lambda row: calc_whip(
                             row[f'bb_{pos}_{team}_{game}'],
                             row[f'h_{pos}_{team}_{game}'],
                             row[f'ip_{pos}_{team}_{game}'],
                         ), axis = 1)
                     
    # drop all original cols 
    df_ops_era_cleaned = df[[col for col in df.columns 
                         if any(metric in col for metric in ['batavg', 
                                                             'obp', 
                                                             'slg', 
                                                             'era', 
                                                             'k9', 
                                                             'whip'])]]

    # add back target                 
    df_ops_era_cleaned['home_outcome'] = home_outcome # y: if home wins = 1, home lose = 0 
    return df_ops_era_cleaned 

def get_n_games_rolling_avg(df,n_games, weights):
    '''
    from df, only select last n games 
    and calculate the rolling average for all the metrics
    '''

    df_only_n = keep_n_prev_games(df,n_games)
    df_summed = calc_sum_stats(df_only_n, weights)
    df_metrics = make_metrics(df_summed)

    return df_metrics


def softmax_with_temperature(logits, temperature=1.0):
    """
    Compute the softmax of logits with a temperature scaling.
    
    Args:
    - logits (np.array): A 1D array or vector of logits (raw scores).
    - temperature (float): The temperature scaling factor. Default is 1.0.
    
    Returns:
    - np.array: The probabilities corresponding to the softmax distribution.
    """
    # Apply temperature scaling
    logits_scaled = logits * temperature
    
    # Compute softmax (numerically stable version)
    exp_logits = np.exp(logits_scaled - np.max(logits_scaled))  # Subtract max for numerical stability
    softmax_probs = exp_logits / np.sum(exp_logits)
    
    return softmax_probs




def main():
    step_1 = True       # Grid serach for hyperparams 
    step_2 = False      # Determine optimal historic games window
    step_3 = False      # Determin optimal temperature coefficient


    # Determine hyperparameters
    if step_1:
        K = 5
        data = pd.read_csv('data_averages/running_162_data.csv', low_memory=False)
        X = data.iloc[:,2:-1]
        y = data.iloc[:,-1:].squeeze()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=142)
        model = LogisticRegression(max_iter=1000)
        
        
        scoring = {
        'accuracy': 'accuracy',       
        'roc_auc': 'roc_auc'           
        }

        param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],       # Regularization strengths
        'penalty': ['l1', 'l2'],            # Regularization types
        'solver': ['saga', 'liblinear']     # Solver types
        }

        grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=K,               
        scoring= scoring,
        refit = 'accuracy',
        verbose=2,          
        n_jobs=-1           
        )

        grid_search.fit(X_train, y_train)

        # Best parameters
        print("Best parameters:", grid_search.best_params_)

        # Best cross-validation score
        print("Best cross-validation score:", grid_search.best_score_)

        # Retrieve accuracy and ROC-AUC for each fold of the best hyperparams
        best_index = grid_search.best_index_

        # Retrieve accuracy and AUC for each fold using best index
        fold_accuracies = [grid_search.cv_results_[f'split{i}_test_accuracy'][best_index] for i in range(K)]
        fold_roc_aucs = [grid_search.cv_results_[f'split{i}_test_roc_auc'][best_index] for i in range(K)]

        # Print fold-specific scores
        print("Accuracy scores for each fold (best hyperparameters):", fold_accuracies)
        print("ROC-AUC scores for each fold (best hyperparameters):", fold_roc_aucs)

        # Best model
        best_model = grid_search.best_estimator_

        # Predict on test data
        y_pred = best_model.predict(X_test)

        # Evaluate accuracy
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Test accuracy:", test_accuracy)
        
        #save grid search results
        with open('grid_search_LR.pkl', 'wb') as f:
            pickle.dump(grid_search, f)

        C = grid_search.best_params_['C']
        penalty = grid_search.best_params_['penalty']
        solve_method = grid_search.best_params_['solver']

        # Get test accuracy
        model = LogisticRegression(C= C, penalty= penalty, solver = solve_method, max_iter=1000)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        fold_acc = accuracy_score(y_test, y_pred)
        print(fold_acc)
        print(roc_auc_score(y_test, y_pred))

    # Using determined hyper params choose window size for prev games
    if step_2:
        K = 5
        kf = KFold(n_splits=K, shuffle=True, random_state=142)
        print("Checking Game windows")
        results = np.zeros([20,K])
        auc_scores = np.zeros([20,K])
        for b in range(14,15):
            print(f"On game {b}")
            full_data = pd.read_csv(f"data_averages/metric_avg_game_{b}.csv")
            full_data.dropna(inplace=True)
            X = full_data.iloc[:,1:-1].to_numpy()
            y = full_data.iloc[:,-1:].to_numpy()

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=142)
            folds = []
            auc = []
            for train_index, val_index in kf.split(X_train):
                X_train_k, X_val = X_train[train_index], X_train[val_index]
                y_train_k, y_val = y_train[train_index], y_train[val_index]
                model = LogisticRegression(C= 1, penalty='l1', solver='liblinear', random_state=142)

                model.fit(X_train_k, y_train_k)

                y_pred = model.predict(X_val)
                fold_acc = accuracy_score(y_val, y_pred)
                folds.append(fold_acc)
                auc.append(roc_auc_score(y_val, y_pred))
            results[b-1] = folds
            auc_scores[b-1] = auc

        print(results.mean(1))
        print(auc_scores.mean(1))
        np.save('LR_games_search_acc_v2.npy', results)
        np.save('LR_games_search_auc_v2.npy', auc_scores)
    
        best_result = np.argmax(results.mean(1))+1
        print(f"Best window size is: {best_result}")
        
        # Get test accuracy
        full_data = pd.read_csv(f"data_averages/metric_avg_game_{best_result}.csv")

        X = full_data.iloc[:,1:-1].to_numpy()
        y = full_data.iloc[:,-1:].to_numpy()

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=142)
        model = LogisticRegression(C = 1, penalty='l1', solver='liblinear', random_state=142)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print(f"Test accuracy: {acc}")
        print(f"Test AUC: {auc}")
    
    # Determine the decay coefficient
    if step_3:
        N = 14 #best_result
        K = 5
        xs = np.linspace(1,N,N)
        temps = np.linspace(0,0.5,100)
        results = np.zeros((len(temps),K))
        auc_scores = np.zeros((len(temps),K))
        df = pd.read_csv('data/full_training_data.csv', low_memory=False)

        
        #df[df.select_dtypes(include=['bool','O']).columns] = df.select_dtypes(include=['bool','O']).astype(int)

        kf = KFold(n_splits=K, shuffle=True, random_state=142)
        scaler = StandardScaler()
        for i,temp in enumerate(temps):
            print(f"Working on {i}")
            folds = []
            auc = []
            weights = softmax_with_temperature(N-xs,temp)
            output = get_n_games_rolling_avg(df, N, weights)
            X = output.iloc[:,:-1]
            y = output.iloc[:,-1:]

            X_train, _, y_train,_ = train_test_split(X, y, test_size=0.3, random_state=142)
            X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
            for train_index, val_index in kf.split(X_train):
                X_train_k, X_val = X_train[train_index], X_train[val_index]
                y_train_k, y_val = y_train[train_index], y_train[val_index]
                
                X_train_k = scaler.fit_transform(X_train_k)
                X_val = scaler.transform(X_val)

                model = LogisticRegression(C= 1, penalty='l1', solver='liblinear', random_state=142)

                model.fit(X_train_k, y_train_k)

                y_pred = model.predict(X_val)
                fold_acc = accuracy_score(y_val, y_pred)
                
                folds.append(fold_acc)
                auc.append(roc_auc_score(y_val, y_pred))
            print(folds)
            results[i] = folds
            auc_scores[i] = auc

        np.save("LR_accuracy_wavg_v2.npy", results)
        np.save("LR_auc_wavg_v2.npy", auc_scores)
        

        best_coef_index = np.argmax(results.mean(1))
        coefficient = temps[best_coef_index]
        print(f"best temperature is: {coefficient}")

        weights = softmax_with_temperature(N-xs,coefficient)
        output = get_n_games_rolling_avg(df, N, weights)
        X = output.iloc[:,:-1]
        y = output.iloc[:,-1:]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=142)
        X_train, y_train = X_train.to_numpy(), y_train.to_numpy()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LogisticRegression(C= 1, penalty='l1', solver='liblinear', random_state=142)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print(f"Test accuracy: {acc}")
        print(f"Test AUC: {auc}")


    


if __name__ == "__main__":
    main()
        
