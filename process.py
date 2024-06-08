import pandas as pd
import numpy as np

df=pd.read_csv('data\lichess_db_puzzle.csv')

lim=500000

df_train=df.iloc[:lim,:][['PuzzleId','FEN','Moves']]
targets=df.iloc[:lim,:]['Rating']


import cupy as cp
import chess
import pandas as pd
import numpy as np

def extract_features_gpu(fen_list):
    try:
        board_data = []

        for fen in fen_list:
            board = chess.Board(fen)
            for piece in board.piece_map().values():
                piece_type = piece.piece_type  # Get the piece type

                # Now you can convert piece_type to a number
                piece_type_num = float(piece_type)
                board_data.append(piece_type_num)  # Add the piece type to the board data

        board_data = cp.array(board_data)  # Transfer data to GPU

        # Example of CUDA kernel for computing material counts
        piece_values = cp.array([1, 3, 3, 5, 9, 0, 1, 3, 3, 5, 9, 0], dtype=cp.float32)
        material_counts = cp.zeros((len(fen_list), 2), dtype=cp.float32)

        # GPU kernel code to compute material counts
        kernel_code = """
        extern "C" __global__ void compute_material(float* board_data, float* material_counts, float* piece_values, int fen_list_length) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < 64 * fen_list_length) {
            int piece = (int)board_data[idx];
            int player = piece / 6;  // White or Black
            int piece_type = piece % 6;  // Pawn, Knight, Bishop, Rook, Queen, King
            atomicAdd(&material_counts[(idx / 64) * 2 + player], piece_values[piece_type]);
        }
    }
    """

        # Compile and run the kernel
        mod = cp.RawModule(code=kernel_code)
        compute_material = mod.get_function("compute_material")

        block_size = 64
        grid_size = (len(fen_list) * 64 + block_size - 1) // block_size
        compute_material((grid_size,), (block_size,), (board_data, material_counts, piece_values))

        # Transfer results back to CPU
        material_counts = cp.asnumpy(material_counts)

        # Process other features...

        features = pd.DataFrame(material_counts, columns=['material_white', 'material_black'])
        features['material_imbalance'] = features['material_white'] - features['material_black']

        # Add other features like king safety, mobility, etc.

        return features

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


fen_list = df_train['FEN'].tolist()
features = extract_features_gpu(fen_list)

# save features
features.to_csv('./features.csv', index=False)