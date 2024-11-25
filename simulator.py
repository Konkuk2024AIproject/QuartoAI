import numpy as np
from multiprocessing import Pool, cpu_count

# from machines_p1 import P1
from jaejin_p1 import P1
from seojin_machine import P2
# from machines_p2 import P2
# from yubin_p2 import P2
import time

TF = (0, 1)
ITERATIONS = 1
result = [0, 0, 0]  # [Player 1 win, Player 2 win, Draw]

def check_group(group):
    if 0 in group:
        return False
    for i in range(4):
        if len(set((x - 1) & (1 << i) for x in group)) == 1:
            return True
    return False

def check_win(board):
    for i in range(4):
        if check_group(board[i]) or check_group(board[:, i]):
            return True
    if check_group([board[i][i] for i in range(4)]) or check_group([board[i][3 - i] for i in range(4)]):
        return True
    for i in range(3):
        for j in range(3):
            if check_group([board[i][j], board[i][j + 1], board[i + 1][j], board[i + 1][j + 1]]):
                return True
    return False

def simulate_game(_):
    players = [P1, P2]
    board = np.zeros((4, 4), dtype=int)
    pieces = [(i, j, k, l) for i in TF for j in TF for k in TF for l in TF]  # All 16 pieces
    available_pieces = pieces[:]

    for turn in range(16):
        p1, p2 = players[turn % 2](board, available_pieces), players[(turn + 1) % 2](board, available_pieces)
        selected_piece = p2.select_piece()
        r, c = p1.place_piece(selected_piece)
        available_pieces.remove(selected_piece)
        board[r][c] = pieces.index(selected_piece) + 1
        if check_win(board):
            return turn % 2
    return 2

if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        results = pool.map(simulate_game, range(ITERATIONS))

    result = [results.count(0), results.count(1), results.count(2)]

    print(f"{ITERATIONS} games played.")
    print(f"P1 win: {result[0]}, P2 win: {result[1]}, Draw: {result[2]}")