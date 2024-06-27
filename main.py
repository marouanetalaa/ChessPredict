import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from preprocess import preprocess_training_data, preprocess_test_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load data
data_path = "lichess_db_puzzle.csv"
data = pd.read_csv(data_path)

# Split data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the split data to new CSV files for preprocessing
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)


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


# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = preprocess_training_data("train_data.csv")
test_loader = preprocess_test_data("test_data.csv")

# Update input_dim to include new features
input_dim = (
    64 + 64 * 64 + 3
)  # 64 for FEN embedding, 64 * 64 for moves embedding, 3 for new features
model = NeuralNetwork(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Initialize best loss as infinity
best_test_loss = float("inf")

# Training loop
model.train()
for epoch in range(10):  # Adjust epochs as necessary
    epoch_loss = 0
    print(f"Starting epoch {epoch+1}")
    for batch_idx, batch in enumerate(train_loader):
        fen = batch["FEN"].to(device)
        moves = batch["Moves"].to(device)
        captures = batch["Captures"].to(device).view(-1, 1)
        sacrifices = batch["Sacrifices"].to(device).view(-1, 1)
        sol_length = batch["SolutionLength"].to(device).view(-1, 1)
        inputs = torch.cat(
            (fen, moves.view(moves.size(0), -1), captures, sacrifices, sol_length),
            dim=1,
        )
        targets = (
            batch["Rating"].to(device).view(-1, 1)
        )  # Convert targets to float for regression

        optimizer.zero_grad()
        outputs = model(inputs)  # No scaling back to original range
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item()}")

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss}")

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
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
            outputs = model(inputs)  # No scaling back to original range
            preds = outputs.cpu().numpy().flatten()
            all_preds.extend(preds)
            targets = batch["Rating"].cpu().numpy().flatten()
            all_targets.extend(targets)

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(test_loader)}")

    test_loss = mean_squared_error(all_targets, all_preds)
    print(f"Epoch {epoch+1} Test Loss: {test_loss}")

    # Save the model if it has the lowest test loss
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), "best_chess_puzzle_model.pth")
        print(f"Model saved with test loss {best_test_loss}")

# Final evaluation on the test set
model.eval()
all_preds = []
all_targets = []
print("Starting final evaluation")
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
        outputs = model(inputs)  # No scaling back to original range
        preds = outputs.cpu().numpy().flatten()
        all_preds.extend(preds)
        targets = batch["Rating"].cpu().numpy().flatten()
        all_targets.extend(targets)

        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(test_loader)}")

# Calculate MSE
mse = mean_squared_error(all_targets, all_preds)
print(f"Mean Squared Error: {mse}")
