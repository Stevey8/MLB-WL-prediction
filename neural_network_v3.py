import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn.init as init
from sklearn.model_selection import KFold
import math
from sklearn.metrics import roc_auc_score


def keep_n_prev_games(df, n_games):
    """
    Filter DataFrame columns to keep only statistics from the n most recent games.

    Args:
        df (pandas.DataFrame): Input DataFrame containing game statistics
        n_games (int): Number of previous games to keep statistics for

    Returns:
        pandas.DataFrame: DataFrame with filtered columns containing only statistics 
            from the n most recent games and any non-game specific columns

    The function identifies columns that end with game numbers (e.g. '_1', '_2') 
    and keeps only those where the game number is <= n_games. It also keeps any
    columns that don't end in a game number.
    """
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



class MultiLayerNetwork(nn.Module):
    def __init__(self, in_sz, h_sz, out_sz, layers, dropout_prob):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_sz, h_sz),
            nn.ReLU(),
            nn.Dropout(dropout_prob) 
        )
        for _ in range(layers):
            self.net.append(nn.Linear(h_sz, h_sz))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(dropout_prob))
        self.net.append(nn.Linear(h_sz, out_sz))
        self.net.append(nn.Sigmoid())

        # Smart parameter initialization to avoid run-time issues in training
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu') 
                if layer.bias is not None:
                    init.zeros_(layer.bias)  # Initialize biases to zeros

    def forward(self, x):
        return self.net(x)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select which steps to run
    step_1 = False      # Used in hyperparam search
    step_2 = False      # Used to find optimal historic games window
    step_3 = True      # Used to determine temperature coefficient
    test_step3 = False  # Used to get testing accuracy of model
    
    if step_1:
        K = 5
        SEED = 42
        df = pd.read_csv('data_averages/running_162_data.csv', low_memory=False)
        X = df.iloc[:,2:-1]
        Y = df.iloc[:,-1]

        df.dropna(inplace=True)
        df[df.select_dtypes(include=['bool','O']).columns] = df.select_dtypes(include=['bool','O']).astype(int)

        kf = KFold(n_splits=K, shuffle=True, random_state=SEED)
        scaler = StandardScaler()

        # Normalize inputs and split data
        X_scaled = scaler.fit_transform(X)
        X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

        # Move to tensors
        X_train = torch.Tensor(X_train_df).to(torch.float32).to(device)
        y_train = torch.Tensor(y_train_df.to_numpy()).to(torch.float32).to(device)

        X_test_tensor = torch.Tensor(X_test_df).to(torch.float32).to(device)
        y_test_tensor = torch.Tensor(y_test_df.to_numpy()).to(torch.float32).to(device)
    
        # Hyper params
                                # Options tested
        BATCH_SIZE = 64
        LR = 1e-4
        EPOCHS = 50
        IN_SZ = X.shape[1]
        H_SZ = 100              #[10,50,100,200,300]
        OUT_SZ = 1
        DROPOUT_PROBs = [0.2]     #[0, 0.05, 0.1, 0.2, 0.5]
        LAYERS = 5              #[1,3,5]

        # Track results
        results = np.zeros([len(DROPOUT_PROBs),K])
        auc_scores = np.zeros([len(DROPOUT_PROBs),K])

        for i, DROPOUT_PROB in enumerate(DROPOUT_PROBs):
            print(i)
            folds = []
            auc = []
            # 5-fold cross validation
            for train_index, val_index in kf.split(X_train):
                    X_train_k, X_val = X_train[train_index], X_train[val_index]
                    y_train_k, y_val = y_train[train_index], y_train[val_index]
                        

                    train = DataLoader(list(zip(X_train_k, y_train_k)), batch_size=BATCH_SIZE, shuffle=True)

                    network = MultiLayerNetwork(IN_SZ, H_SZ, OUT_SZ, LAYERS, DROPOUT_PROB).to(device)
                    criterion = nn.BCELoss()
                    optimizer = optim.Adam(network.parameters(), lr = LR)

                    for epoch in range(EPOCHS):
                        print(f"     {epoch}")
                        for j, (x, y) in enumerate(train):
                            logits = network(x).squeeze()
                            loss = criterion(logits, y)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # Test model
                    network.eval() # Set to evaluation mode
                    with torch.no_grad():
                        test_logits = network(X_val).squeeze()
                        predictions = (test_logits >= 0.5).int()
                        accuracy = (predictions == y_val).float().mean()
                        folds.append(accuracy.item())

                        auc.append(roc_auc_score(y_val.cpu().numpy(), test_logits.cpu().numpy()))

            results[i] = folds
            auc_scores[i] = auc

        # Store results
        np.save('Baseline_search_acc.npy', results)
        np.save('Baseline_search_auc.npy', auc_scores)
        return

    # Determine the historic game window size
    if step_2:
        print("step 2")
        K = 5
        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        scaler = StandardScaler()

        results = np.zeros([20,K])
        auc_scores = np.zeros([20,K])

        for i in range(1,21):
            folds = []
            auc = []

            df = pd.read_csv(f"data_averages/metric_avg_game_{i}.csv")
            df.dropna(inplace=True)
            df[df.select_dtypes(include=['bool','O']).columns] = df.select_dtypes(include=['bool','O']).astype(int)
            
            X = df.iloc[:,1:-1]
            Y = df.iloc[:,-1:]

            # Hyper params from previous step
            SEED = 42
            BATCH_SIZE = 64
            LR = 1e-4
            EPOCHS = 50
            IN_SZ = X.shape[1]
            H_SZ = 100 
            OUT_SZ = 1
            DROPOUT_PROB = 0.2 
            LAYERS = 5
            torch.manual_seed(SEED)
                    
            X_scaled = scaler.fit_transform(X)
            X_train_df, _, y_train_df, _ = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

            X_train = torch.Tensor(X_train_df).to(torch.float32).to(device)
            y_train = torch.Tensor(y_train_df.to_numpy()).to(torch.float32).to(device)

            # 5-fold cross validation
            for train_index, val_index in kf.split(X_train):
                X_train_k, X_val = X_train[train_index], X_train[val_index]
                y_train_k, y_val = y_train[train_index], y_train[val_index]
                    

                train = DataLoader(list(zip(X_train_k, y_train_k)), batch_size=BATCH_SIZE, shuffle=True)

                network = MultiLayerNetwork(IN_SZ, H_SZ, OUT_SZ, LAYERS, DROPOUT_PROB).to(device)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(network.parameters(), lr = LR)

                for epoch in range(EPOCHS):
                    print(epoch)
                    for j, (x, y) in enumerate(train):
                        logits = network(x)
                        loss = criterion(logits, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Validate model
                network.eval() # Set to evaluation mode
                with torch.no_grad():
                    test_logits = network(X_val)
                    predictions = (test_logits >= 0.5).int()
                    accuracy = (predictions == y_val).float().mean()
                    folds.append(accuracy.item())
                    auc.append(roc_auc_score(y_val.cpu(), test_logits.cpu()))

            results[i-1] = folds
            auc_scores[i-1] = auc
            print("Accuracy: {results}")
            print("AUC scores: {auc_scores}")
        
        
        np.save('NN_val_acc_games_search.npy', results)
        np.save('NN_val_auc_games_search.npy', auc_scores)
        return

    if step_3:
        print("step 3")
        K = 5
        N = 18
        M = 100
        time_stamps = np.linspace(1,N,N)
        temps = [0, 0.05, 0.1]#np.linspace(0,0.5,M)
        results = np.zeros((M,K))
        auc_scores = np.zeros([M,K])

        df = pd.read_csv('data/full_training_data.csv', low_memory=False)

        df.dropna(inplace=True)
        df[df.select_dtypes(include=['bool','O']).columns] = df.select_dtypes(include=['bool','O']).astype(int)

        kf = KFold(n_splits=K, shuffle=True, random_state=42)
        scaler = StandardScaler()
        


        scaler = StandardScaler()
        results = np.zeros([100,K])
        auc_scores = np.zeros([100,K])

        # Test temperature coefficient
        for i,temp in enumerate(temps):
            folds = []
            auc = []
            weights = softmax_with_temperature(np.array(N-time_stamps),temp)
            output = get_n_games_rolling_avg(df, N, weights)

            print(f"Working on {i}")

            X = output.iloc[:,:-1]
            Y = output.iloc[:,-1:]

            SEED = 42
            BATCH_SIZE = 64
            LR = 1e-4
            EPOCHS = 75
            IN_SZ = X.shape[1]
            H_SZ = 100
            OUT_SZ = 1
            DROPOUT_PROB = 0.2
            LAYERS = 5
            torch.manual_seed(SEED)
                    
            X_scaled = scaler.fit_transform(X)
            X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

            X_train = torch.Tensor(X_train_df).to(torch.float32).to(device)
            y_train = torch.Tensor(y_train_df.to_numpy()).to(torch.float32).to(device).squeeze()

            print(y_train.shape)

            X_test_tensor = torch.Tensor(X_test_df).to(torch.float32).to(device)
            y_test_tensor = torch.Tensor(y_test_df.to_numpy()).to(torch.float32).to(device).squeeze()

            for train_index, val_index in kf.split(X_train):
                X_train_k, X_val = X_train[train_index], X_train[val_index]
                y_train_k, y_val = y_train[train_index], y_train[val_index]
                    

                train = DataLoader(list(zip(X_train_k, y_train_k)), batch_size=BATCH_SIZE, shuffle=True)

                network = MultiLayerNetwork(IN_SZ, H_SZ, OUT_SZ, LAYERS, DROPOUT_PROB).to(device)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(network.parameters(), lr = LR)

                for epoch in range(EPOCHS):
                    print(epoch)
                    for j, (x, y) in enumerate(train):
                        logits = network(x).squeeze()
                        loss = criterion(logits, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Test model
                network.eval() # Set to evaluation mode
                with torch.no_grad():
                    test_logits = network(X_val).squeeze()
                    predictions = (test_logits >= 0.5).int()
                    accuracy = (predictions == y_val).float().mean()
                    folds.append(accuracy.item())
                    auc.append(roc_auc_score(y_val.cpu(), test_logits.cpu()))

            results[i] = folds
            auc_scores[i] = auc

            print("Accuracy: {results}")
            print("AUC scores: {auc_scores}")
        
        np.save('NN_val_acc_temp_search_v3.npy', results)
        np.save('NN_val_auc_temp_search_v3.npy', auc_scores)

        print(results)
        return

    if test_step3:
        N = 18
        M = 1
        time_stamps = np.linspace(1,N,N)
        temps = [0.15]#np.linspace(0,0.5,M)


        df = pd.read_csv('data/full_training_data.csv', low_memory=False)
        df.dropna(inplace=True)
        df[df.select_dtypes(include=['bool','O']).columns] = df.select_dtypes(include=['bool','O']).astype(int)

        scaler = StandardScaler()

        for i,temp in enumerate(temps):
            folds = []
            auc = []
            weights = softmax_with_temperature(np.array(N-time_stamps),temp)
            output = get_n_games_rolling_avg(df, N, weights)

            print(f"Working on {i}")

            X = output.iloc[:,:-1]
            Y = output.iloc[:,-1:]

            SEED = 42
            BATCH_SIZE = 64
            LR = 1e-4
            EPOCHS = 300
            #REPORT = 250
            IN_SZ = X.shape[1]
            H_SZ = 100
            OUT_SZ = 1
            DROPOUT_PROB = 0.2
            LAYERS = 5
            torch.manual_seed(SEED)
                    
            X_scaled = scaler.fit_transform(X)
            X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

            X_train = torch.Tensor(X_train_df).to(torch.float32).to(device)
            y_train = torch.Tensor(y_train_df.to_numpy()).to(torch.float32).to(device).squeeze()

            X_test_tensor = torch.Tensor(X_test_df).to(torch.float32).to(device)
            y_test_tensor = torch.Tensor(y_test_df.to_numpy()).to(torch.float32).to(device).squeeze()

            
            train = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)

            network = MultiLayerNetwork(IN_SZ, H_SZ, OUT_SZ, LAYERS, DROPOUT_PROB).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(network.parameters(), lr = LR)

            for epoch in range(EPOCHS):
                print(epoch)
                for j, (x, y) in enumerate(train):
                    logits = network(x).squeeze()
                    loss = criterion(logits, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Test model
            network.eval() # Set to evaluation mode
            with torch.no_grad():
                test_logits = network(X_test_tensor).squeeze()
                predictions = (test_logits >= 0.5).int()
                accuracy = (predictions == y_test_tensor).float().mean()
                print(f"Test Accuracy: {accuracy}")
                auc = roc_auc_score(y_test_tensor.cpu(), test_logits.cpu())
                print(f"Test AUC: {auc}")

if __name__ == "__main__":
    main()
