import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


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
    self.net.append(nn.Softmax())


  def forward(self, x):
    return self.net(x)
  
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyper parameters
    
    SEED = 42
    BATCH_SIZE = 64
    LR = 0.00001
    EPOCHS = 50
    REPORT = 100
    IN_SZ = 200
    H_SZ = 250
    OUT_SZ = 2
    DROPOUT_PROB = 0.5
    LAYERS = 5
    torch.manual_seed(SEED)
    

    # Load data
    data_dir = "data/"
    full_data = pd.read_csv(f"{data_dir}full_training_data_ops_era.csv")
    full_data[full_data.select_dtypes(include='bool').columns] = full_data.select_dtypes(include='bool').astype(int)
    
    
    full_data.dropna(inplace=True)
    
    X = full_data.iloc[:,1:-1]
    y = full_data.iloc[:,-1:]

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.Tensor(X_train_df.to_numpy())#.to(device)
    y_train = torch.Tensor(y_train_df.to_numpy()).squeeze().long()#.to(device)

    train = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)

    network = MultiLayerNetwork(IN_SZ, H_SZ, OUT_SZ, LAYERS, DROPOUT_PROB)#.to(device)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch}")
        for i, (x, y) in enumerate(train):
            logits = network(x)
            probs = F.log_softmax(logits, dim=-1)
            loss = F.nll_loss(probs, y)
            loss.backward()
            if i % REPORT == 0:
                print(loss.detach(), end=" ")

            for param in network.parameters():
                param.data = param.data - LR * param.grad
            for param in network.parameters():
                param.grad.zero_()
    # Save model
    torch.save(network.state_dict(), f"{data_dir}model.pt")

    # Test model
    network.eval() # Set to evaluation mode
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_df.values)#.to(device)
        y_test_tensor = torch.LongTensor(y_test_df.values.ravel())#.to(device)
        
        test_logits = network(X_test_tensor)
        test_probs = F.softmax(test_logits, dim=-1)
        predictions = torch.argmax(test_probs, dim=1)
        
        accuracy = (predictions == y_test_tensor).float().mean()
        print(f"\nTest Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
   main()
