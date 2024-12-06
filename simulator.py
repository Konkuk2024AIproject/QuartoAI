import random
import numpy as np
import time
from multiprocessing import Pool, cpu_count

# from machines_yeonwook import P1
# # 7일때=> P1 win: 0, P2 win: 50, Draw: 0

# from machines_v3 import P1
# # 5일떄=> P1 win: 8, P2 win: 28, Draw: 14

from machines_refactor import P2

from machines_refactor_7 import P1
# P1 win: 10, P2 win: 18, Draw: 22

# from machines_v2_deep_filter_optimized import P1
# # 7일때=> P1 win: 12, P2 win: 13, Draw: 25
# # 6일때=> P1 win: 16, P2 win: 13, Draw: 21
# # 5일때=> P1 win: 11, P2 win: 13, Draw: 26

# from machines_mcts import P1
# # 7일때=> P1 win: 2, P2 win: 39, Draw: 9
# # 6일때=> P1 win: 4, P2 win: 37, Draw: 9
# # 5일때=> P1 win: 4, P2 win: 39, Draw: 7

# from minmax_jih import P1
# # 7일때=> P1 win: 13, P2 win: 37, Draw: 0
# # 6일때=> P1 win: 10, P2 win: 39, Draw: 1
# # 5일때=> P1 win: 1, P2 win: 47, Draw: 2

TF = (0, 1)
ITERATIONS = 50


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
    random.seed(time.time())

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


# def main():
#     res = simulate_game(0)
#     rs = 'Player 1 wins' if res == 0 else 'Player 2 wins' if res == 1 else 'Draw'
#     print(f"Result: {rs}")


# if __name__ == "__main__":
#     cProfile.run('main()')


def run():
    result = [0, 0, 0]
    for game in range(ITERATIONS):
        print(f"Game {game + 1} started.")
        res = simulate_game(game)
        rst = 'Player 1 wins' if res == 0 else 'Player 2 wins' if res == 1 else 'Draw'
        print(f"Result: {rst}")
        result[res] += 1
    return result


def run_multiprocessing():
    with Pool(cpu_count()) as pool:
        results = pool.map(simulate_game, range(ITERATIONS))
    result = [results.count(0), results.count(1), results.count(2)]
    return result


def show_result(r):
    print(f"{ITERATIONS} games played.")
    print(f"P1 win: {r[0]}, P2 win: {r[1]}, Draw: {r[2]}")


if __name__ == "__main__":
    show_result(run_multiprocessing())
