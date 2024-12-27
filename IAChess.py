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
# 2. Modelo de red neuronal
# ===========================
model = Sequential([
    Input(shape=(8, 8, 12)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Cargar pesos preentrenados (si existen)
# model.load_weights("ruta_a_tus_pesos.h5")

# ===========================
# 3. Evaluación del tablero
# ===========================
def evaluate_board(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    evaluation = 0
    for piece in chess.PIECE_TYPES:
        evaluation += len(board.pieces(piece, chess.WHITE)) * piece_values[piece]
        evaluation -= len(board.pieces(piece, chess.BLACK)) * piece_values[piece]
    return evaluation

def evaluate_position_with_nn(board):
    tensor = board_to_tensor(board).reshape(1, 8, 8, 12)
    score = model.predict(tensor)[0][0]
    return score

# ===========================
# 4. Algoritmo Minimax
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
# 5. Selección del mejor movimiento
# ===========================
def find_best_move(board, depth1, depth2, nn_model):
    move_scores = []
    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth1, float('-inf'), float('inf'), False, evaluate_board)
        tactical_score = evaluate_position_with_nn(board)
        move_scores.append((move, score, tactical_score))
        board.pop()
    prioritized_moves = sorted(move_scores, key=lambda x: x[2], reverse=True)[:5]
    best_move = None
    best_score = float('-inf')
    for move, _, _ in prioritized_moves:
        board.push(move)
        deep_score = minimax(board, depth2, float('-inf'), float('inf'), False, evaluate_board)
        board.pop()
        if deep_score > best_score:
            best_score = deep_score
            best_move = move
    return best_move

# ===========================
# 6. Ejecución interactiva
# ===========================
def show_menu():
    print("\nOpciones:")
    print("1. Realizar un movimiento.")
    print("2. Deshacer el último movimiento.")
    print("3. Reiniciar el tablero.")
    print("4. Salir.")
    choice = input("Elige una opción (1/2/3/4): ").strip()
    return choice

if __name__ == "__main__":
    # Preguntar al usuario por la posición inicial
    fen = input("Introduce la posición inicial en formato FEN (deja vacío para la posición inicial estándar): ").strip()
    if fen:
        try:
            board = chess.Board(fen)
        except ValueError:
            print("FEN inválido. Usando posición inicial estándar.")
            board = chess.Board()
    else:
        board = chess.Board()

    # Elegir el color de la IA
    ia_color = input("¿Deseas que la IA juegue con blancas (b) o negras (n)? ").strip().lower()
    ia_plays_white = ia_color == "b"

    # Configurar profundidades
    depth1 = 2  # Profundidad inicial (rápida)
    depth2 = 4  # Profundidad avanzada (detallada)

    # Historial de movimientos
    move_history = []

    # Bucle de juego
    while not board.is_game_over():
        print("\nTablero actual:\n")
        print(board)

        # Mostrar el menú de opciones
        choice = show_menu()

        if choice == '1':  # Realizar un movimiento
            user_action = input("Ingresa tu movimiento (en notación SAN o UCI): ").strip().lower()
            if len(user_action) == 4:
                move = chess.Move.from_uci(user_action)  # Notación UCI
            else:
                try:
                    move = board.parse_san(user_action)  # Notación SAN
                except ValueError:
                    print("Movimiento inválido. Intenta nuevamente.")
                    continue  # Volver a mostrar el menú

            if move in board.legal_moves:
                board.push(move)
                move_history.append(move)
            else:
                print("Movimiento ilegal. Intenta nuevamente.")

        elif choice == '2':  # Deshacer el último movimiento
            if move_history:
                last_move = move_history.pop()
                board.pop()
                print(f"Movimiento deshecho: {last_move}")
            else:
                print("No hay movimientos para deshacer.")

        elif choice == '3':  # Reiniciar el tablero
            board = chess.Board()  # Reiniciar el tablero
            move_history.clear()  # Limpiar el historial de movimientos
            print("Juego reiniciado.")

        elif choice == '4':  # Salir
            print("¡Gracias por jugar!")
            break

        # Si es el turno de la IA, hacer el movimiento
        if board.turn == chess.WHITE and ia_plays_white or board.turn == chess.BLACK and not ia_plays_white:
            print("Pensando en el mejor movimiento para la IA...")
            best_move = find_best_move(board, depth1, depth2, model)
            board.push(best_move)
            move_history.append(best_move)
            print(f"La IA juega: {best_move}")

    # Fin del juego
    print("\n¡Juego terminado!")
    print(f"Resultado: {board.result()}")