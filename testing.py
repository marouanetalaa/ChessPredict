import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# Basic feature extraction from FEN
def extract_material_balance(fen):
    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9,
                    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9}
    balance = sum(piece_values.get(piece, 0) for piece in fen.split()[0])
    return balance

def extract_move_count(moves):
    return len(moves.split())

# Define the best trial parameters
best_trial_params = {
    'n_layers': 1,
    'n_units_l0': 119,
    'dropout_l0': 0.028735330507266718,
    'learning_rate': 0.0838905920459873,
    'batch_size': 46
}

# Define BestTrial class
class BestTrial:
    def __init__(self, params):
        self.params = params

    def suggest_int(self, name, low, high):
        return self.params[name]

    def suggest_float(self, name, low, high):
        return self.params[name]

# Define Neural Network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, trial):
        super(NeuralNetwork, self).__init__()
        layers = []
        n_layers = trial.suggest_int('n_layers', 1, 5)
        for i in range(n_layers):
            input_dim = input_dim if i == 0 else n_units
            n_units = trial.suggest_int(f'n_units_l{i}', 4, 128)
            layers.append(nn.Linear(input_dim, n_units))
            layers.append(nn.ReLU())
            dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.0, 0.5)
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(n_units, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def load_model(model_path, input_dim, best_trial):
    model = NeuralNetwork(input_dim, best_trial).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_test_data(test_data_path, scaler):
    test_data = pd.read_csv(test_data_path)
    test_data['MaterialBalance'] = test_data['FEN'].apply(extract_material_balance)
    test_data['MoveCount'] = test_data['Moves'].apply(extract_move_count)
    features = ['MaterialBalance', 'MoveCount']
    X_test = test_data[features]
    X_test_scaled = scaler.transform(X_test)
    return torch.tensor(X_test_scaled, dtype=torch.float32)

def test_model(model_path, test_data_path, scaler, best_trial):
    X_test = preprocess_test_data(test_data_path, scaler)
    model = load_model(model_path, X_test.shape[1], best_trial)
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy().flatten()
    print("Predictions on Test Data:", y_pred)
    return y_pred

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the training data to fit the scaler
data_path = 'lichess_db_puzzle_50000.csv'
data = pd.read_csv(data_path)
data['MaterialBalance'] = data['FEN'].apply(extract_material_balance)
data['MoveCount'] = data['Moves'].apply(extract_move_count)
features = ['MaterialBalance', 'MoveCount']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set best trial parameters
best_trial = BestTrial(best_trial_params)

# Test data path and model path
test_data_path = 'test_data_set.csv'  # Path to your test data
model_path = 'best_chess_puzzle_model.pth'  # Path to the saved model
# Call the test function
y_pred = test_model(model_path, test_data_path, scaler, best_trial)

#save to txt with each line containing the integer format of the prediction only 2282 lines
with open('predictions.txt', 'w') as f:
    start = 0
    end = 2282
    for pred in y_pred:
        f.write(str(int(pred)) + '\n')
        start += 1
        if start == end:
            break
