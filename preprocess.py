import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re


class ChessDataset(Dataset):
    def __init__(self, dataframe, include_rating=True):
        self.data = dataframe
        self.fen_list = self.data["FEN"].tolist()
        self.move_list = self.data["Moves"].tolist()
        self.include_rating = include_rating
        if include_rating:
            self.rating_list = self.data["Rating"].tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen = self.fen_list[idx]
        moves = self.move_list[idx]

        fen_embedding = self.fen_to_matrix(fen)
        move_embedding = self.moves_to_sequence(moves)
        captures = count_captures(moves)
        sacrifices = count_sacrifices(moves, fen)
        sol_length = solution_length(moves)

        sample = {
            "FEN": fen_embedding,
            "Moves": move_embedding,
            "Captures": captures,
            "Sacrifices": sacrifices,
            "SolutionLength": sol_length,
        }
        if self.include_rating:
            sample["Rating"] = self.rating_list[idx]

        return sample

    def fen_to_matrix(self, fen):
        piece_dict = {
            "p": 1,
            "r": 2,
            "n": 3,
            "b": 4,
            "q": 5,
            "k": 6,
            "P": 7,
            "R": 8,
            "N": 9,
            "B": 10,
            "Q": 11,
            "K": 12,
        }
        matrix = np.zeros((8, 8), dtype=int)
        rows = fen.split()[0].split("/")

        for i, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    col_idx += int(char)
                else:
                    matrix[i, col_idx] = piece_dict[char]
                    col_idx += 1
        return matrix.flatten()

    def moves_to_sequence(self, moves):
        move_list = moves.split()
        move_sequence = []
        for move in move_list:
            move_sequence.append(self.move_to_vector(move))
        move_sequence = np.array(move_sequence)
        move_sequence = self.pad_sequence(move_sequence, 64)
        return move_sequence.flatten()

    def move_to_vector(self, move):
        move = re.sub("[^\w]", "", move)  # Remove special characters
        vector = np.zeros(64, dtype=int)
        if len(move) == 4:
            start_pos = self.square_to_index(move[:2])
            end_pos = self.square_to_index(move[2:])
            vector[start_pos] = 1
            vector[end_pos] = 2
        return vector

    def square_to_index(self, square):
        file_dict = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
        rank_dict = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
        return rank_dict[square[1]] * 8 + file_dict[square[0]]

    def pad_sequence(self, sequence, length):
        if sequence.shape[0] < length:
            padding = np.zeros(
                (length - sequence.shape[0], sequence.shape[1]), dtype=int
            )
            sequence = np.vstack((sequence, padding))
        return sequence


def count_captures(moves):
    return moves.count("x")


def count_sacrifices(moves, fen):
    piece_values = {
        "P": 1,
        "N": 3,
        "B": 3,
        "R": 5,
        "Q": 9,
        "K": 0,
        "p": 1,
        "n": 3,
        "b": 3,
        "r": 5,
        "q": 9,
        "k": 0,
    }
    board = fen.split()[0]
    moves_list = moves.split()
    sacrifices = 0

    for move in moves_list:
        if "x" in move:
            capturing_piece = move[0] if move[0].isalpha() else "P"
            captured_piece = board[board.index(move[-2:]) - 1]
            if piece_values[capturing_piece] >= piece_values[captured_piece]:
                sacrifices += 1

    return sacrifices


def solution_length(moves):
    return len(moves.split())


def preprocess_training_data(file_path):
    df = pd.read_csv(file_path)
    dataset = ChessDataset(df, include_rating=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    return dataloader


def preprocess_test_data(file_path):
    df = pd.read_csv(file_path)
    dataset = ChessDataset(df, include_rating=True)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    return dataloader


def collate_fn(batch):
    fens = torch.tensor([item["FEN"] for item in batch], dtype=torch.float32)
    moves = torch.tensor([item["Moves"] for item in batch], dtype=torch.float32)
    captures = torch.tensor([item["Captures"] for item in batch], dtype=torch.float32)
    sacrifices = torch.tensor(
        [item["Sacrifices"] for item in batch], dtype=torch.float32
    )
    sol_lengths = torch.tensor(
        [item["SolutionLength"] for item in batch], dtype=torch.float32
    )

    collated_batch = {
        "FEN": fens,
        "Moves": moves,
        "Captures": captures,
        "Sacrifices": sacrifices,
        "SolutionLength": sol_lengths,
    }
    if "Rating" in batch[0]:
        ratings = torch.tensor([item["Rating"] for item in batch], dtype=torch.float32)
        collated_batch["Rating"] = ratings

    return collated_batch
