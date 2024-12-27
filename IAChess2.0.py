import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

# ===========================
# 1. Representación del tablero
# ===========================
def board_to_tensor(board):
    """
    Convierte un tablero de ajedrez en una representación de tensor 8x8x12.
    """
    matrix = np.zeros((8, 8, 12))
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        matrix[row, col, piece.piece_type - 1] = 1 if piece.color == chess.WHITE else -1
    return matrix / 1  # Normalización

# ===========================
# 2. Matriz de importancia de las casillas
# ===========================
square_importance = np.array([
    [0.5, 0.6, 0.6, 0.7, 0.7, 0.6, 0.6, 0.5],
    [0.6, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 0.6],
    [0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0.6],
    [0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7],
    [0.7, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7],
    [0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0.6],
    [0.6, 0.7, 0.7, 0.8, 0.8, 0.7, 0.7, 0.6],
    [0.5, 0.6, 0.6, 0.7, 0.7, 0.6, 0.6, 0.5],
])

# ===========================
# 3. Modelo de red neuronal
# ===========================
model = Sequential([
    Input(shape=(8, 8, 12)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ===========================
# 4. Evaluación del tablero
# ===========================
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.2,
    chess.BISHOP: 3.3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def evaluate_board_with_coefficients(board):
    """
    Evalúa el tablero considerando los valores de las piezas y la importancia estratégica de las casillas.
    """
    evaluation = 0
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        base_value = piece_values[piece.piece_type]
        coefficient = square_importance[row, col]
        value = base_value * coefficient
        evaluation += value if piece.color == chess.WHITE else -value
    return evaluation

def evaluate_position_with_nn(board):
    tensor = board_to_tensor(board).reshape(1, 8, 8, 12)
    score = model.predict(tensor)[0][0]
    return score

# ===========================
# 5. Algoritmo Minimax
# ===========================
def minimax(board, depth, alpha, beta, maximizing_player, evaluation_function):
    if depth == 0 or board.is_game_over():
        return evaluation_function(board)
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, evaluation_function)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, evaluation_function)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# ===========================
# 6. Selección del mejor movimiento
# ===========================
def find_best_move(board, depth1, depth2, nn_model):
    move_scores = []
    for move in board.legal_moves:
        board.push(move)
        
        # Evaluar usando Minimax con coeficientes
        score = minimax(board, depth1, float('-inf'), float('inf'), False, evaluate_board_with_coefficients)
        
        # Evaluar la posición con la red neuronal
        tactical_score = evaluate_position_with_nn(board)
        
        # Combinar las evaluaciones
        combined_score = score + tactical_score
        move_scores.append((combined_score, move))
        
        board.pop()
    
    # Ordenar movimientos por puntuación combinada
    best_move = max(move_scores, key=lambda x: x[0])[1]
    return best_move

# ===========================
# 7. Ejecución interactiva
# ===========================
if __name__ == "__main__":
    board = chess.Board()
    depth1 = 3
    depth2 = 5
    
    while not board.is_game_over():
        print("\nTablero actual:\n")
        print(board)

        if board.turn == chess.WHITE:
            print("Pensando en el mejor movimiento para las blancas...")
            best_move = find_best_move(board, depth1, depth2, model)
            board.push(best_move)
            print(f"Blancas juegan: {best_move}")
        else:
            user_move = input("Introduce tu movimiento (notación SAN): ")
            try:
                board.push_san(user_move)
            except ValueError:
                print("Movimiento inválido. Intenta de nuevo.")
