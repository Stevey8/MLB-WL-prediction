import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

class BaseballGamePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BaseballGamePredictor, self).__init__()
        
        # LSTM layer to process the sequences of past 5 games (5 time steps, 10 features for each game pair)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer after the LSTM to make the prediction
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)  # Output layer for binary classification (win/loss)
        
        # Sigmoid activation to output probabilities
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step for classification
        out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through the fully connected layers
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        
        # Apply sigmoid activation
        out = self.sigmoid(out)
        
        return out




def main():
    data_dir = "data/"
    full_data = pd.read_csv(f"{data_dir}full_training_data.csv")
    full_data[full_data.select_dtypes(include='bool').columns] = full_data.select_dtypes(include='bool').astype(int)

    X = full_data.iloc[:,:-1]
    y = full_data.iloc[:,-1:]

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.Tensor(X_train_df.to_numpy()).to(device)
    y_train = torch.Tensor(y_train_df.to_numpy()).squeeze().long().to(device)

    train = DataLoader(list(zip(X_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyper parameters
    # Hyperparameters
    input_size = 10  # 5 features per team * 2 teams
    hidden_size = 64  # You can experiment with this
    num_layers = 1  # Number of LSTM layers
    learning_rate = 0.001
    num_epochs = 20  # Number of training epochs

    # Model, loss function, and optimizer
    model = BaseballGamePredictor(input_size, hidden_size, num_layers)
    criterion = nn.BCELoss()  # Binary Cross-Entropy loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train:
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs.squeeze(), targets)  # .squeeze() to match target shape
            running_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted = (outputs.squeeze() > 0.5).float()  # Convert to 0 or 1 based on probability threshold
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


if __name__ == "__main__":
   main()
