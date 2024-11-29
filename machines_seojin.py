import pickle
import numpy as np
import random
import time
from collections import defaultdict

Piece = tuple[int, int, int, int]

inf = 1e9
time_limit = [5, 5, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 15, 15, 5, 5]
# production = True


class AI:
    def __init__(self, board: np.ndarray, available_pieces: list[Piece], is_first: bool):
        self.board = board - 1  # 2D array representing the board (-1: empty, 0~15: pieces)
        self.available_pieces = available_pieces  # Tuple of available pieces (e.g., (1, 0, 1, 0))
        self.is_first = is_first  # True if the AI is player 1, False if player 2

        self.turn = 17 - len(available_pieces)  # Current turn number

        # load pre-trained model
        # try:
        #     with open('data.pickle', 'rb') as f:
        #         data = pickle.load(f)
        #         self.visits = data['visits']
        #         self.wins = data['wins']
        # except FileNotFoundError:

        # Initialize MCTS variables
        self.visits = defaultdict(int)
        self.wins = defaultdict(int)

    def select_piece(self) -> Piece:
        print(f'Selecting piece for turn {self.turn}')

        if self.turn == 1:
            print('first turn: return arbitrary piece')
            print()
            return self.available_pieces[0]

        self.mcts()
        parent_visit = self.visits[self.to_state(self.board, -1)]

        best_score = -inf
        best_piece = None
        for piece in self.available_pieces:
            pidx = self.to_piece_index(piece)
            state = self.to_state(self.board, pidx)
            if (score := self.visits[state]) > best_score:
                best_score = score
                best_piece = piece
            print(f'{piece}: {score:.3f} ({self.wins[state]} / {self.visits[state]})')
        print('best =', best_piece)
        print()
        return best_piece

    def place_piece(self, selected_piece: Piece):
        print(f'Placing piece for turn {self.turn}, selected piece = {selected_piece}')

        pidx = self.to_piece_index(selected_piece)
        self.mcts(selected_piece)
        parent_visit = self.visits[self.to_state(self.board, pidx)]

        best_score = -inf
        best_move = None
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == -1:
                    self.board[row][col] = pidx
                    state = self.to_state(self.board, -1)
                    if (score := self.visits[state]) > best_score:
                        best_score = score
                        best_move = (row, col)

                    print(f'({row}, {col}): {score:.3f} ({self.wins[state]} / {self.visits[state]})')
                    self.board[row][col] = -1

        print('best =', best_move)
        print()

        return best_move

    def mcts(self, selected_piece: None | Piece = None):
        start = time.time()
        c = 0
        while time.time() - start < time_limit[self.turn - 1]:
            for _ in range(100):
                self.simulate(selected_piece)
            c += 100

        print(f"Turn {self.turn} MCTS finished in {time.time() - start:.3f}s with {c} simulations")
        print()
        # save model
        # if not production:
        #     with open('data.pickle', 'wb') as f:
        #         pickle.dump({'visits': self.visits, 'wins': self.wins}, f)

    @staticmethod
    def to_state(board, selected_piece: int):
        return tuple((*board.ravel().tolist(), selected_piece))

    @staticmethod
    def to_piece_index(piece: Piece):
        return sum(1 << (3 - i) for i, x in enumerate(piece) if x)

    def get_score(self, state, parent_visit: int, is_me: bool):
        expect = self.wins[state] / (self.visits[state] + 1)
        explored = np.sqrt(2 * (1 + np.log(parent_visit + 1)) / (self.visits[state] + 1))
        if is_me:
            return expect + 2 * explored
        else:
            return 1 - expect + 2 * explored

    def simulate(self, selected_piece: None | Piece):
        board = self.board.copy()
        available_pieces = list(self.available_pieces)
        history = []
        parent_visit = 0

        history.append(self.to_state(board, self.to_piece_index(selected_piece) if selected_piece else -1))

        # break when not visited state
        while True:
            # print('simulate', board, selected_piece, end='\n\n')
            turn = 17 - len(available_pieces)
            if turn > 16:
                break
            if selected_piece:
                is_me = turn % 2 == self.is_first
                piece = self.to_piece_index(selected_piece)
                state = self.to_state(board, piece)
                if self.visits[state] == 0:
                    break
                best_score = -inf
                best_move = None
                positions = [(i, j) for i in range(4) for j in range(4) if board[i, j] == -1]
                random.shuffle(positions)
                for row, col in positions:
                    board[row][col] = piece
                    if self.check_win_by_move(board, (row, col)):
                        best_move = (row, col)
                        break
                    score = self.get_score(s := self.to_state(board, -1), parent_visit, is_me)
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)
                    # print((row, col), score, (self.wins[s], self.visits[s]))
                    board[row][col] = -1
                # print('best =', best_move)
                board[best_move] = piece
                history.append(self.to_state(board, -1))
                available_pieces.remove(selected_piece)
                selected_piece = None

                if self.check_win_by_move(board, best_move):
                    break
            else:
                is_me = turn % 2 != self.is_first
                state = self.to_state(board, -1)
                if self.visits[state] == 0:
                    break
                best_score = -inf
                best_move = None
                for piece in available_pieces:
                    pidx = self.to_piece_index(piece)
                    score = self.get_score(s := self.to_state(board, pidx), parent_visit, is_me)
                    if score > best_score:
                        best_score = score
                        best_move = piece
                piece = best_move
                selected_piece = piece
                history.append(self.to_state(board, self.to_piece_index(piece)))
            parent_visit = self.visits[state]

        while True:
            turn = 17 - len(available_pieces)
            if turn > 16:
                break
            if selected_piece:
                pidx = self.to_piece_index(selected_piece)
                empty_positions = [(i, j) for i in range(4) for j in range(4) if board[i, j] == -1]
                random.shuffle(empty_positions)
                candidate = []
                for row, col in empty_positions:
                    board[row, col] = pidx
                    if self.check_win_by_move(board, (row, col)):
                        break
                    else:
                        candidate.append((row, col))
                    board[row, col] = -1
                else:
                    row, col = random.choice(empty_positions if not candidate else candidate)
                    board[row, col] = pidx
                history.append(self.to_state(board, -1))
                available_pieces.remove(selected_piece)
                selected_piece = None

                if self.check_win_by_move(board, (row, col)):
                    break
            else:
                selected_piece = random.choice(available_pieces)
                history.append(self.to_state(board, self.to_piece_index(selected_piece)))

        end_turn = 16 - len(available_pieces)
        score = .5 if not self.check_win(board) else (1 if end_turn % 2 == self.is_first else 0)

        # print('score =', score)
        for state in history:
            # print(state)
            self.visits[state] += 1
            self.wins[state] += score

    def is_terminal(self, board, last_move=None):
        # Check for terminal state (win or full board)
        win = self.check_win_by_move(board, last_move) if last_move else self.check_win(board)
        return win or not np.any(board == -1)

    def check_win(self, board):
        if any(self.is_winning_set(board[i, :]) for i in range(4)):
            return True
        if any(self.is_winning_set(board[:, j]) for j in range(4)):
            return True

        if self.is_winning_set(np.diag(board)):
            return True
        if self.is_winning_set(np.diag(np.fliplr(board))):
            return True

        for i in range(3):
            for j in range(3):
                subgrid = board[i:i+2, j:j+2]
                if self.is_winning_set(subgrid.ravel()):
                    return True
        return False

    def check_win_by_move(self, board, last_move):
        row, col = last_move

        if self.is_winning_set(board[row, :]):
            return True
        if self.is_winning_set(board[:, col]):
            return True

        if row == col and self.is_winning_set(np.diag(board)):
            return True

        if row + col == 3 and self.is_winning_set(np.diag(np.fliplr(board))):
            return True

        # Check for surrounding 2x2 square
        for i in range(max(0, row - 1), min(2, row) + 1):
            for j in range(max(0, col - 1), min(2, col) + 1):
                subgrid = board[i:i+2, j:j+2]
                if self.is_winning_set(subgrid.ravel()):
                    return True
        return False

    def is_winning_set(self, line: np.ndarray):
        if -1 in line:
            return False
        for i in range(4):
            if len(set(x & (1 << i) for x in line)) == 1:
                return True
        return False


# Declare instances
P1 = lambda board, available_pieces: AI(board, available_pieces, True)
P2 = lambda board, available_pieces: AI(board, available_pieces, False)
