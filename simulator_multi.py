import logging
import random
import numpy as np
import time
import uuid
from itertools import permutations
# noinspection PyPackageRequirements,PyProtectedMember
from pathos.multiprocessing import ProcessingPool as Pool

from machines_yeonwook import P1
from machines_yeonwook2 import P2

TF = (0, 1)

ITERATIONS = 1
PLAYERS = {
    # 'v2_deep_filter': (machines_v2_deep_filter.P1, machines_v2_deep_filter.P2),
    # 'v2_deep_filter_optimized': (machines_v2_deep_filter_optimized.P1, machines_v2_deep_filter_optimized.P2),
    'yw': (P1, P1),
    'yw2': (P2, P2),
}


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


def format_piece(piece):
    if piece == 0:
        return '----'
    return f"{piece-1:04b}"


def simulate_game(players):
    n1 = players[0].__name__
    n2 = players[1].__name__
    uid = str(uuid.uuid4())[:8]
    logger = logging.getLogger(f"game-{uid}")
    logger.addHandler(logging.FileHandler(f"logs/game-{n1}-{n2}-{uid}.log"))
    logger.setLevel(logging.INFO)

    logger.info(f"Game started: {n1} vs {n2}")
    random.seed(time.time())

    board = np.zeros((4, 4), dtype=int)
    pieces = [(i, j, k, l) for i in TF for j in TF for k in TF for l in TF]  # All 16 pieces
    available_pieces = pieces[:]

    for turn in range(16):
        logger.info(f"Turn {turn + 1}")
        p1, p2 = players[turn % 2](board, available_pieces), players[(turn + 1) % 2](board, available_pieces)
        selected_piece = p2.select_piece()
        r, c = p1.place_piece(selected_piece)
        available_pieces.remove(selected_piece)
        board[r][c] = pieces.index(selected_piece) + 1
        for row in board:
            logger.info(' '.join(map(format_piece, row)))
        if check_win(board):
            logger.info(f"Game ended at turn {turn + 1}. Winner: Player {turn % 2 + 1}")
            print(f"Game ended at turn {turn + 1}. Winner: Player {turn % 2 + 1}")
            return turn % 2
    logger.info("Game ended at turn 16 without a winner.")
    print("Game ended at turn 16 without a winner.")
    return 2


# def main():
#     res = simulate_game(0)
#     rs = 'Player 1 wins' if res == 0 else 'Player 2 wins' if res == 1 else 'Draw'
#     print(f"Result: {rs}")


# if __name__ == "__main__":
#     cProfile.run('main()')


def run(players):
    result = [0, 0, 0]
    for game in range(ITERATIONS):
        print(f"Game {game + 1} started.")
        res = simulate_game(*players)
        rst = 'Player 1 wins' if res == 0 else 'Player 2 wins' if res == 1 else 'Draw'
        print(f"Result: {rst}")
        result[res] += 1
    return result


def run_multiprocessing(players):
    with Pool() as pool:
        results = pool.map(simulate_game, [players] * ITERATIONS)
    result = [results.count(0), results.count(1), results.count(2)]
    return result


def run_league():
    print(f"League started with {ITERATIONS} games")
    print(f"Players: {', '.join(PLAYERS)}")
    results = {player: [0, 0, 0] for player in PLAYERS}
    for k1, k2 in permutations(PLAYERS, 2):
        s = [PLAYERS[k1][0], PLAYERS[k2][1]]
        r = run_multiprocessing(s)
        print(f"{k1} vs {k2}: {r[0]} - {r[2]} - {r[1]}")
        for i in range(3):
            results[k1][i] += r[i]
            results[k2][[1, 0, 2][i]] += r[i]
    print("Results:")
    for k, v in results.items():
        print(f"{k}: {v[0]} - {v[2]} - {v[1]} (total: {v[0] + v[2] * 0.5})")


if __name__ == "__main__":
    run_league()
