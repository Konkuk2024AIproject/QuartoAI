import numpy as np
import random
from collections import defaultdict

inf = 1e9


class AI:
    def __init__(self, board: np.ndarray, available_pieces: list[tuple[int, int, int, int]], is_first: bool):
        self.board = board - 1  # 2D array representing the board (-1: empty, 0~15: pieces)
        self.available_pieces = available_pieces  # Tuple of available pieces (e.g., (1, 0, 1, 0))
        self.is_first = is_first  # True if the AI is player 1, False if player 2

        self.turn = 16 - len(available_pieces)  # Current turn number

        self.visits = defaultdict(int)  # Number of visits for each state-action pair
        self.wins = defaultdict(int)  # Number of wins for each state-action pair

    def select_piece(self) -> tuple[int, int, int, int]:
        if self.turn == 1:
            return self.available_pieces[0]

        self.mcts()
        parent_visit = self.visits[self.to_state(self.board, -1)]

        best_score = -inf
        best_piece = None
        for piece in self.available_pieces:
            pidx = self.to_piece_index(piece)
            if (score := self.get_score(self.to_state(self.board, pidx), parent_visit)) > best_score:
                best_score = score
                best_piece = piece
        return best_piece

    def place_piece(self, selected_piece):
        pidx = self.to_piece_index(selected_piece)

        self.mcts(selected_piece)
        parent_visit = self.visits[self.to_state(self.board, pidx)]

        best_score = -inf
        best_move = None
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == -1:
                    self.board[row][col] = pidx
                    if (score := self.get_score(self.to_state(self.board, None), parent_visit)) > best_score:
                        best_score = score
                        best_move = (row, col)
        return best_move

    def mcts(self, selected_piece: None | tuple[int, int, int, int] = None, iterations=10000):
        for _ in range(iterations):
            self.simulate(selected_piece)

    @staticmethod
    def to_state(board, selected_piece):
        return tuple((*board.ravel(), selected_piece))

    @staticmethod
    def to_piece_index(piece):
        return sum(1 << (3 - i) for i, x in enumerate(piece) if x)

    def get_score(self, state, parent_visit):
        if self.visits[state] == 0:
            return inf
        return self.wins[state] / self.visits[state] + 1.41 * (2 * np.log(parent_visit) / self.visits[state]) ** .5

    def simulate(self, selected_piece):
        board = self.board.copy()
        available_pieces = list(self.available_pieces)
        history = []
        parent_visit = 0

        # break when not visited state
        while not self.is_terminal(board):
            if selected_piece:
                piece = self.to_piece_index(selected_piece)
                state = self.to_state(board, piece)
                if state not in self.visits:
                    break
                best_score = -inf
                best_move = None
                for row in range(4):
                    for col in range(4):
                        if board[row, col] == -1:
                            board[row, col] = piece
                            score = self.get_score(self.to_state(board, -1), parent_visit)
                            if score > best_score:
                                best_score = score
                                best_move = (row, col)
                            board[row, col] = -1
                board[best_move] = piece
                history.append(self.to_state(board, -1))
            else:
                state = self.to_state(board, -1)
                if state not in self.visits:
                    break
                best_score = -inf
                best_move = None
                for piece in available_pieces:
                    pidx = self.to_piece_index(piece)
                    score = self.get_score(self.to_state(board, pidx), parent_visit)
                    if score > best_score:
                        best_score = score
                        best_move = piece
                piece = best_move
                available_pieces.remove(piece)
                selected_piece = piece
                history.append(self.to_state(board, piece))
            parent_visit = self.visits[state]

        # random simulation
        while not self.is_terminal(board):
            if selected_piece:
                pidx = self.to_piece_index(selected_piece)
                empty_positions = list(zip(*np.where(board == -1)))
                row, col = random.choice(empty_positions)
                board[row, col] = pidx
                history.append(self.to_state(board, -1))
            else:
                piece = random.choice(available_pieces)
                pidx = self.to_piece_index(piece)
                available_pieces.remove(piece)
                empty_positions = list(zip(*np.where(board == -1)))
                row, col = random.choice(empty_positions)
                board[row, col] = pidx
                history.append(self.to_state(board, pidx))

        result = self.get_result(board)
        end_turn = 16 - len(available_pieces)
        score = .5 if result == 0 else (1 if end_turn % 2 == self.is_first else 0)
        for state in history:
            self.visits[state] += 1
            self.wins[state] += score

    def is_terminal(self, board):
        # Check for terminal state (win or full board)
        return self.check_win(board) or not np.any(board == -1)

    def get_result(self, board):
        if self.check_win(board):
            return 1
        return 0

    def check_win(self, board):
        for row in board:
            if self.is_winning_set(row):
                return True
        for col in board.T:
            if self.is_winning_set(col):
                return True
        if self.is_winning_set(np.diag(board)):
            return True
        if self.is_winning_set(np.diag(np.fliplr(board))):
            return True
        # check 2x2 subgrids
        for i in range(3):
            for j in range(3):
                if self.is_winning_set(board[i:i + 2, j:j + 2].ravel()):
                    return True
        return False

    def is_winning_set(self, line):
        if -1 in line:
            return False
        for i in range(4):
            if len(set(x & (1 << i) for x in line)) == 1:
                return True
        return False


P1 = lambda board, available_pieces: AI(board, available_pieces, True)
P2 = lambda board, available_pieces: AI(board, available_pieces, False)
