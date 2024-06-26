import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load data
data_path = 'lichess_db_puzzle_50000.csv'
data = pd.read_csv(data_path)

# Basic feature extraction from FEN
def extract_material_balance(fen):
    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9,
                    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9}
    balance = sum(piece_values.get(piece, 0) for piece in fen.split()[0])
    return balance

def extract_move_count(moves):
    return len(moves.split())

data['MaterialBalance'] = data['FEN'].apply(extract_material_balance)
data['MoveCount'] = data['Moves'].apply(extract_move_count)

# Select relevant features and target
features = ['MaterialBalance', 'MoveCount']
X = data[features]
y = data['Rating']  # Using Rating as the target variable

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define Neural Network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, params):
        super(NeuralNetwork, self).__init__()
        layers = []
        n_layers = params['n_layers']
        input_dim = input_dim
        for i in range(n_layers):
            n_units = params[f'n_units_l{i}']
            layers.append(nn.Linear(input_dim, n_units))
            layers.append(nn.ReLU())
            dropout_rate = params[f'dropout_l{i}']
            layers.append(nn.Dropout(dropout_rate))
            input_dim = n_units
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Best parameters
best_params = {
    'n_layers': 1,
    'n_units_l0': 119,
    'dropout_l0': 0.028735330507266718,
    'learning_rate': 0.0838905920459873,
    'batch_size': 46
}

# Train the best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = NeuralNetwork(X_train.shape[1], best_params).to(device)
learning_rate = best_params['learning_rate']
optimizer = optim.Adam(best_model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

best_model.train()
for epoch in range(1000):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = best_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Evaluate the model
best_model.eval()
with torch.no_grad():
    y_pred = best_model(X_test).cpu().numpy().flatten()
mse = mean_squared_error(y_test.cpu().numpy(), y_pred)
print(f'Mean Squared Error: {mse}')

# Save the best model
torch.save(best_model.state_dict(), 'best_chess_puzzle_model.pth')
