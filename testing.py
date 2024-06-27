import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from prepo import preprocess_test_data


# Define Neural Network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        layers = [
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),  # Output layer for regression
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = (
    64 + 64 * 64 + 3
)  # 64 for FEN embedding, 64 * 64 for moves embedding, 3 for new features
model = NeuralNetwork(input_dim).to(device)
model.load_state_dict(torch.load("best_chess_puzzle_model.pth"))
model.eval()

# Preprocess the test data
test_loader = preprocess_test_data("test_data_set.csv")

# Generate predictions
all_preds = []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        fen = batch["FEN"].to(device)
        moves = batch["Moves"].to(device)
        captures = batch["Captures"].to(device).view(-1, 1)
        sacrifices = batch["Sacrifices"].to(device).view(-1, 1)
        sol_length = batch["SolutionLength"].to(device).view(-1, 1)
        inputs = torch.cat(
            (fen, moves.view(moves.size(0), -1), captures, sacrifices, sol_length),
            dim=1,
        )
        outputs = model(inputs)
        preds = outputs.cpu().numpy().flatten()
        all_preds.extend(preds)

# Save predictions to txt
with open("predictions.txt", "w") as f:
    for pred in all_preds[:2282]:  # Ensure we only write the first 2282 predictions
        f.write(str(int(pred)) + "\n")

print("Predictions saved to predictions.txt")
