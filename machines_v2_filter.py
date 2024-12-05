import logging

import numpy as np
import random
import time
from collections import defaultdict
from numpy.typing import NDArray

PieceType = tuple[int, int, int, int]
BoardType = NDArray[int]
PositionType = tuple[int, int]

# handlers = [logging.FileHandler(f'log_{int(time.time())}.logs')]
# handlers = [logging.StreamHandler(), logging.FileHandler(f'log_{int(time.time())}.logs')]
handlers = []
# handlers = [logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, handlers=handlers)
logger = logging.getLogger(__name__)


class StateType:
    unvisited = 0
    visited = 1
    terminal = 2


class AI:
    """
    Class of AI player. this class is instantiated with
    current state of the board and the list of available pieces.

    attributes:
    - board: the current state of the board
    - available_pieces: the list of available pieces
    - is_first: whether the AI is the first player
    """

    def __init__(self, board: BoardType, available_pieces: list[PieceType], is_first: bool):
        self.board = board - 1  # 0-indexed, -1=empty
        self.available_pieces = available_pieces
        self.is_first = is_first
        self.selected_piece: PieceType | None = None
        self.turn: int = 16 - len(available_pieces) + 1

        self.status = defaultdict(int)
        self.parent = defaultdict(tuple)
        self.children = defaultdict(list)
        self.visits = defaultdict(int)
        self.wins = defaultdict(int)

    def select_piece(self) -> PieceType:
        """
        Select a piece from the list of available pieces.

        return:
        - piece: the selected piece
        """
        logger.info(f'Selecting piece for turn {self.turn}')

        if self.turn == 1:
            logger.info('First turn: selecting arbitrary piece')
            return random.choice(self.available_pieces)

        self.build()

        state = self.to_state(self.board, None)
        best_child = None
        best_score = -1
        for child in self.children[state]:
            score = (self.wins[child]) / (self.visits[child] + 1)
            logger.info(f'{self.to_piece(child[-1])}: {score:.3f} ({self.wins[child]}/{self.visits[child]})')
            if score > best_score:
                best_child = child
                best_score = score
        *_, pidx = best_child
        logger.info(f'Selected piece {self.to_piece(pidx)}')
        return self.to_piece(pidx)

    def place_piece(self, piece: PieceType) -> PositionType:
        """
        Place the piece on the board.

        arguments:
        - piece: the piece to place

        return:
        - position: the position to place the piece (row, column)
        """
        logger.info(f'Placing piece {piece} on the board on turn {self.turn}')

        self.selected_piece = piece
        self.build()

        pi = self.to_index(piece)
        best_move = None
        best_score = -1
        for r in range(4):
            for c in range(4):
                if self.board[r][c] == -1:
                    self.board[r][c] = pi
                    child = self.to_state(self.board, None)
                    self.board[r][c] = -1
                    score = (self.wins[child]) / (self.visits[child] + 1)
                    logger.info(f'{(r, c)}: {score:.3f} ({self.wins[child]}/{self.visits[child]})')
                    if score > best_score:
                        best_move = (r, c)
                        best_score = score
        logger.info(f'Placed piece {piece} at {best_move}')
        return best_move

    def build(self):
        """
        Build the monte carlo tree.
        """
        logger.info('Building Monte Carlo tree.')
        started_at = time.time()
        cnt = 0
        while time.time() - started_at < 15:
            for _ in range(1000):
                self.simulate()
            cnt += 1
        logger.info(f'Finished building tree after {cnt} iterations.')

    def make_children(self, state: tuple[int], board: BoardType, pieces: list[PieceType],
                      selected_piece: PieceType | None):
        """
        Make children of the current state.

        arguments:
        - state: the current state
        - board: the board
        - pieces: the list of available pieces
        - selected_piece: the selected piece
        """
        res = []
        if selected_piece:
            pi = self.to_index(selected_piece)
            for r in range(4):
                for c in range(4):
                    if board[r][c] == -1:
                        board[r][c] = pi
                        res.append(self.to_state(board, None))
                        board[r][c] = -1
        else:
            for piece in pieces:
                pi = self.to_index(piece)
                res.append(self.to_state(board, pi))

        logger.debug(f'Generated children for state {state}.')
        for s in res:
            logger.debug(f'Child: {s}')
        self.children[state] = res

    def evaluate(self, state: tuple[int], is_me: bool) -> float:
        """
        Evaluate the state.

        arguments:
        - state: the state
        - is_me: whether the AI is the player

        return:
        - score: the score
        """

        score = self.wins[state] if is_me else (self.visits[state] - self.wins[state])
        expected = score / (self.visits[state] + 1)
        parent_visit = self.visits[self.parent[state]]
        explored = np.sqrt(2 * (np.log(parent_visit + 1) + 1) / (self.visits[state] + 1))
        return expected + 2 * explored

    def check_win_condition(self, board: BoardType) -> list[list[bool]]:
        """
        Check properties to win the game.

        arguments:
        - board: the board

        return:
        - win_condition: the win condition
        """
        cond = [[False, False] for _ in range(4)]

        groups = [
            *(board[i] for i in range(4)),
            *(board[:, i] for i in range(4)),
            np.diag(board),
            np.diag(np.fliplr(board)),
            *(board[i:i + 2, j:j + 2].ravel() for i in range(3) for j in range(3))
        ]

        v = [*map(self.get_group_win_condition, groups)]
        for l in v:
            for i, b in l:
                cond[i][b] = True
        return cond

    def simulate(self):
        """
        Simulate a single game.
        """
        board = self.board.copy()
        pieces = self.available_pieces.copy()
        selected_piece = self.selected_piece

        history = []
        logger.debug(f'Simulating game with selected piece {selected_piece}')

        parent_state = None
        # run until unvisited node
        state = self.to_state(board, self.to_index(selected_piece) if selected_piece else None)
        while pieces:
            logger.debug(f'Simulating state {state}')
            history.append(state)
            if self.status[state] == StateType.unvisited:
                win = self.check_win(board)
                self.status[state] = StateType.terminal if win else StateType.visited
                self.parent[state] = parent_state
                self.make_children(state, board, pieces, selected_piece)
                break
            elif self.status[state] == StateType.terminal:
                break

            me = self.is_me(pieces, selected_piece)
            best_child = max(
                self.children[state], key=lambda x: self.evaluate(x, me)
            )
            *bd, pi = best_child
            if selected_piece:
                board = np.array(bd).reshape(4, 4)
                pieces.remove(selected_piece)
                selected_piece = None
            else:
                selected_piece = self.to_piece(pi)
            state = best_child
            parent_state = state

        win = self.check_win(board)
        # simulate random moves
        while pieces and not win:
            logger.debug(f'Simulating random move with selected piece {selected_piece}')
            if selected_piece:
                pieces.remove(selected_piece)
                empty_positions = [(i, j) for i in range(4) for j in range(4) if board[i][j] == -1]
                random.shuffle(empty_positions)

                pi = self.to_index(selected_piece)
                for r, c in empty_positions:
                    board[r][c] = pi
                    if self.check_win_by_move(board, (r, c)):
                        win = True
                        break
                    board[r][c] = -1
                else:
                    r, c = random.choice(empty_positions)
                    board[r][c] = pi
                    if self.check_win_by_move(board, (r, c)):
                        win = True
                selected_piece = None
            else:
                win_condition = self.check_win_condition(board)
                for p in pieces:
                    for i in range(4):
                        if win_condition[i][p[i]]:
                            break
                    else:
                        selected_piece = p
                        break
                else:
                    selected_piece = random.choice(pieces)

        turn = 16 - len(pieces)
        result = .5 if not win else 1 if self.is_first == (turn % 2) else 0
        logger.debug(f'Simulation result: {result}, ended at turn {turn}')
        for state in history:
            self.visits[state] += 1
            self.wins[state] += result

    def is_me(self, pieces: list[PieceType], selected_piece: PieceType | None) -> bool:
        """
        Returns whether the AI is the player.
        """
        return (len(pieces) + bool(selected_piece)) % 2 == self.is_first

    @staticmethod
    def to_state(board: BoardType, selected_index: int | None) -> tuple:
        """
        Convert the board and selected piece to a state.

        arguments:
        - board: the board
        - selected_index: the index of the selected piece

        return:
        - state: the state
        """

        return *board.ravel().tolist(), selected_index

    @staticmethod
    def to_index(piece: PieceType) -> int:
        """
        Convert a piece to an index.

        arguments:
        - piece: the piece

        return:
        - index: the index
        """
        return piece[0] * 8 + piece[1] * 4 + piece[2] * 2 + piece[3]

    @staticmethod
    def to_piece(index: int) -> PieceType:
        """
        Convert an index to a piece.

        arguments:
        - index: the index

        return:
        - piece: the piece
        """
        return index // 8, index % 8 // 4, index % 4 // 2, index % 2

    @staticmethod
    def check_win_by_move(board: BoardType, move: tuple) -> bool:
        """
        Check if the board is a winning state after a move.

        arguments:
        - board: the board
        - move: the move

        return:
        - win: whether the board is a winning state
        """
        row, col = move
        if AI.is_group_won(board[row]) or AI.is_group_won(board[:, col]):
            return True
        if row == col and AI.is_group_won(np.diag(board)):
            return True
        if row == 3 - col and AI.is_group_won(np.diag(np.fliplr(board))):
            return True

        # check for surrounding 2x2 squares
        if row > 0 and col > 0 and AI.is_group_won(board[row - 1:row + 1, col - 1:col + 1].ravel()):
            return True
        if row > 0 and col < 3 and AI.is_group_won(board[row - 1:row + 1, col:col + 2].ravel()):
            return True
        if row < 3 and col > 0 and AI.is_group_won(board[row:row + 2, col - 1:col + 1].ravel()):
            return True
        if row < 3 and col < 3 and AI.is_group_won(board[row:row + 2, col:col + 2].ravel()):
            return True

    @staticmethod
    def check_win(board: BoardType) -> bool:
        """
        Check if the board is a winning state.

        arguments:
        - board: the board

        return:
        - win: whether the board is a winning state
        """
        for i in range(4):
            if AI.is_group_won(board[i]) or AI.is_group_won(board[:, i]):
                return True
        if AI.is_group_won(np.diag(board)) or AI.is_group_won(np.diag(np.fliplr(board))):
            return True
        for i in range(3):
            for j in range(3):
                if AI.is_group_won(board[i:i + 2, j:j + 2].ravel()):
                    return True
        return False

    @staticmethod
    def get_group_win_condition(group: NDArray[int]) -> list[tuple[int, bool]]:
        """
        Get the win condition of a group.

        arguments:
        - group: the group

        return:
        - win_condition: the win condition
        """
        g = [x for x in group if x != -1]
        res = []
        for i in range(4):
            v = 1 << i
            if len(set(x & v for x in g)) == 1:
                res.append((i, bool(g[0] & v)))
        return res

    @staticmethod
    def is_group_won(group: NDArray[int]) -> bool:
        """
        Check if a group is a winning group.

        arguments:
        - group: the group

        return:
        - win: whether the group is a winning group
        """
        if -1 in group:
            return False
        return any(len(set(x & v for x in group)) == 1 for v in (1, 2, 4, 8))


def get_instance(is_first: bool):
    """
    Returns a closure that returns an instance of AI.

    arguments:
    - is_first: whether the AI is the first player
    """
    def instance(board: BoardType, available_pieces: list[PieceType]) -> AI:
        return AI(board, available_pieces, is_first)

    return instance


P1 = get_instance(True)
P2 = get_instance(False)
