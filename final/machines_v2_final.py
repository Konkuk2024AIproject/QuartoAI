import copy

import numpy as np
import random
import time
from collections import defaultdict
from itertools import product

PieceType = tuple[int, int, int, int]
PositionType = tuple[int, int]
TL = (20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 24, 18, 12, 6)


class StateType:
    unvisited = 0
    visited = 1
    terminal = 2


class Submachine:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = copy.deepcopy(board) # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.eval_board = np.zeros((4, 4), dtype=int)
        self.eval_piece = np.zeros(len(self.pieces))
        self.row_sym = []
        self.row_eval = []
        self.col_sym = []
        self.col_eval = []
        self.cross_sym = []
        self.cross_eval = []
        self.subgrid_sym = []
        self.subgrid_eval = []

    def select_piece(self):
        if self.board.max():
            self.evaluate_piece()
            while 1:
                best_piece = self.pieces[self.eval_piece.argmax()]
                if best_piece in self.available_pieces:
                    return best_piece
                else:
                    self.eval_piece[self.eval_piece.argmax()] -= 500
        else:
            return random.choice(self.available_pieces)

    def place_piece(self, selected_piece):
        # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        # Available locations to place the piece
        if self.board.max():
            pos = self.evaluate_position(selected_piece)
            if pos != 16:
                while 1:
                    row = pos // 4
                    col = pos % 4
                    if (row, col) in available_locs:
                        return row, col
                    else:
                        self.eval_board[row][col] -= 500
                        pos = np.argmax(self.eval_board)
            else :
                return random.choice(available_locs)
        else:
            return 1,1

    def check_possibility(self, line):
        line = [grid for grid in line if grid != 0]
        characteristics = np.array([self.pieces[idx - 1] for idx in line])
        sym, quantity = [],[]
        if len(characteristics) == 0:
            for i in range(4):
                sym.append(-1)
                quantity.append(0)
        else :
            for char in characteristics.T:
                u_val, count = np.unique(char, return_counts=True)
                if len(u_val) == 2:
                    sym.append(-1)
                    quantity.append(-1)
                else:
                    sym.append(u_val[0])
                    quantity.append(count[0])
        return sym, quantity

    def member_reset(self):
        self.row_sym = []
        self.row_eval = []
        self.col_sym = []
        self.col_eval = []
        self.cross_sym = []
        self.cross_eval = []
        self.subgrid_sym = []
        self.subgrid_eval = []

    def check_possibilities(self):

        self.member_reset()
        for col in range(4):
            sym, quantity = self.check_possibility([self.board[row][col] for row in range(4)])
            if len(sym)!= 0 and len(quantity) != 0:
                self.col_sym.append(sym)
                self.col_eval.append(quantity)
            else:
                self.col_sym.append([-1,-1,-1,-1])
                self.col_eval.append([0,0,0,0])

        for row in range(4):
            sym, quantity = self.check_possibility([self.board[row][col] for col in range(4)])
            if len(sym)!= 0 and len(quantity) != 0:
                self.row_sym.append(sym)
                self.row_eval.append(quantity)
            else:
                self.row_sym.append([-1,-1,-1,-1])
                self.row_eval.append([0,0,0,0])

        sym, quantity = self.check_possibility([self.board[i][i] for i in range(4)])
        if len(sym)!= 0 and len(quantity) != 0:
            self.cross_sym.append(sym)
            self.cross_eval.append(quantity)
        else:
            self.cross_sym.append([-1,-1,-1,-1])
            self.cross_eval.append([0,0,0,0])
        sym, quantity = self.check_possibility([self.board[i][3 - i] for i in range(4)])
        if len(sym)!= 0 and len(quantity) != 0:
            self.cross_sym.append(sym)
            self.cross_eval.append(quantity)
        else:
            self.cross_sym.append([-1,-1,-1,-1])
            self.cross_eval.append([0,0,0,0])

        # Check 2x2 sub-grids
        for r in range(3):
            for c in range(3):
                subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                sym, quantity = self.check_possibility(subgrid)
                if len(sym)!= 0 and len(quantity) != 0:
                    self.subgrid_sym.append(sym)
                    self.subgrid_eval.append(quantity)
                else:
                    self.subgrid_sym.append([-1,-1,-1,-1])
                    self.subgrid_eval.append([0,0,0,0])

    def evaluate_piece(self):
        self.check_possibilities()
        max_vals = [max([max(reval) for reval in self.row_eval]), max([max(ceval) for ceval in self.col_eval]), max([max(creval) for creval in self.cross_eval]), max([max(seval) for seval in self.subgrid_eval])]
        win_conditions_eval = [self.row_eval, self.col_eval, self.cross_eval, self.subgrid_eval]
        win_conditions_sym = [self.row_sym, self.col_sym, self.cross_sym, self.subgrid_sym]

        for c_count in range(4):
            if 3 in np.ravel(np.array(win_conditions_eval[c_count])):
                for sindex, cond in enumerate(win_conditions_sym[c_count]):
                    for idx, sym in enumerate(cond) :
                        if sym != -1 and win_conditions_eval[c_count][sindex][idx] == 3:
                            valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                            valid_piece_set = set(valid_piece) # valid_piece를 set으로 변환
                            result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                            self.eval_piece -= np.array(result_array)

            if 2 in np.ravel(np.array(win_conditions_eval[c_count])):
                for sindex, cond in enumerate(win_conditions_sym[c_count]):
                    for idx, sym in enumerate(cond) :
                        if sym != -1 and win_conditions_eval[c_count][sindex][idx] == 2:
                            valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                            valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                            result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])
                            weight = np.sum(result_array)
                            self.eval_piece += (weight*np.array(result_array))

        if 1 in max_vals:
            for c_count in range(4):
                for sindex, cond in enumerate(win_conditions_sym[c_count]):
                    for idx, sym in enumerate(cond):
                        if sym != -1 and win_conditions_eval[c_count][sindex][idx] == 1:
                            valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                            valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                            result_array = np.array([2 if piece in valid_piece_set else 0 for piece in self.pieces])
                            self.eval_piece += np.array(result_array)

    def calculate_evaluation_value(self, r, c, selected_piece):
        pos_list = []
        self.board[r][c] = self.pieces.index(selected_piece) + 1
        _, count = self.check_possibility([self.board[r][col] for col in range(4)])
        pos_list.append(max(count))
        _, count = self.check_possibility([self.board[row][c] for row in range(4)])
        pos_list.append(max(count))
        _, count = self.check_possibility([self.board[i][i] for i in range(4)])
        pos_list.append(max(count))
        _, count = self.check_possibility([self.board[i][3-i] for i in range(4)])
        pos_list.append(max(count))

        for sg_r in range(3):
            for sg_c in range(3):
                subgrid = [self.board[sg_r][sg_c], self.board[sg_r][sg_c+1], self.board[sg_r+1][sg_c], self.board[sg_r+1][sg_c+1]]
                _, count = self.check_possibility(subgrid)
                pos_list.append(max(count))
        self.board[r][c] = 0
        return pos_list

    @staticmethod
    def check_position(win_conditions_eval, same_value):
        max_positions = []
        for cond_idx, win_condition in enumerate(win_conditions_eval):
            max_positions.append([(cond_idx, linenum, symidx) for linenum, line in enumerate(win_condition) for symidx, val in enumerate(line) if val == same_value])
        return max_positions

    def update_eval_board_2(self, symidx, locs, selected_piece):
        for (r,c) in locs:
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece)) # 3이 되는 부분이 있는지 확인
            if recalculated_max_val == 3:
                for r_sym in self.row_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if piece[symidx] == r_sym[symidx]:
                            piece_make_4s.append(piece)
                            self.eval_board[r][c] -= (len(piece_make_4s))
                for c_sym in self.col_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if piece[symidx] == c_sym[symidx]:
                            piece_make_4s.append(piece)
                            self.eval_board[r][c] -= (len(piece_make_4s))
                for cr_sym in self.cross_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if piece[symidx] == cr_sym[symidx]:
                            piece_make_4s.append(piece)
                            self.eval_board[r][c] -= (len(piece_make_4s))
                for sg_sym in self.subgrid_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if piece[symidx] == sg_sym[symidx]:
                            piece_make_4s.append(piece)
                            self.eval_board[r][c] -= (len(piece_make_4s))
            else:
                self.eval_board[r][c] += 3

    def update_eval_board_1(self, locs, selected_piece):
        for (r,c) in locs:
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece))
            if recalculated_max_val == 2:
                self.eval_board[r][c] += 1
            else:
                self.eval_board[r][c] += 2

    def evaluate_position(self, selected_piece):
        self.check_possibilities()
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        win_conditions_eval = [self.row_eval, self.col_eval, self.cross_eval, self.subgrid_eval]
        win_conditions_sym = [self.row_sym, self.col_sym, self.cross_sym, self.subgrid_sym]

        max_vals = []
        for win_condition in win_conditions_eval:
            max_vals.append(max(np.ravel(np.array(win_condition))))

        for r in range(1,3):
            for c in range(1,3):
                self.eval_board[r][c] += 1
        r, c = 1, 1
        for loc in available_locs:
            r, c = loc
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece))

            if recalculated_max_val == 4:
                return 4*r + c

        if 3 in np.ravel([item for sublist in win_conditions_eval for item in sublist]):
            max_positions = self.check_position(win_conditions_eval, 3) # 3인 곳을 찾아줌
            for cond in max_positions:
                if len(cond)!= 0:
                    for max_position in cond:
                        cond_idx, linenum, symidx = max_position
                        locs = [(r,c) for (r,c) in available_locs if r==linenum]
                        for (r,c) in locs:
                            self.board[r][c] = self.pieces.index(selected_piece) + 1
                            pos_list = self.check_position(win_conditions_eval, 3) # board가 변경되었을 때 다시 3인 곳을 찾아줌
                            for pos_cond in pos_list:
                                if len(pos_cond)!=0:
                                    for pos in pos_cond:
                                        new_cond_idx, new_linenum, new_symidx = pos
                                        if symidx == new_symidx:
                                            sym_to_check = win_conditions_sym[cond_idx][linenum][symidx]
                                            if \
                                            win_conditions_sym[new_cond_idx][new_linenum][new_symidx] != sym_to_check:
                                                self.eval_board[r][c] -= 80
                                            else:
                                                possible_pieces = [piece for piece in self.available_pieces if piece[new_symidx] != sym_to_check]
                                                self.eval_board[r][c] += len(possible_pieces)
                            self.board[r][c] = 0

        if 2 in np.ravel([item for sublist in win_conditions_eval for item in sublist]):
            max_positions = self.check_position(win_conditions_eval, 2)

            for cond in max_positions:
                if len(cond)!= 0:
                    for max_position in cond:
                        cond_idx, linenum, symidx = max_position
                        if cond_idx == 0: # row
                            locs = [(r,c) for (r,c) in available_locs if (r==linenum and self.board[r][c] == 0)]
                            self.update_eval_board_2(symidx, locs, selected_piece)

                        elif cond_idx == 1: # col
                            locs = [(r,c) for (r,c) in available_locs if (c==linenum and self.board[r][c] == 0)]
                            self.update_eval_board_2(symidx, locs, selected_piece)

                        elif cond_idx == 2: # cross
                            locs = [(i,i) for i in range(4) if (i==linenum and self.board[r][c] == 0)]
                            self.update_eval_board_2(symidx, locs, selected_piece)
                            locs = [(i,3-i) for i in range(4) if (i==linenum and self.board[r][c] == 0)]
                            self.update_eval_board_2(symidx, locs, selected_piece)

                        else: # subgrids
                            if linenum == 0:
                                locs = [(i,j) for i in range(2) for j in range(2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                            elif linenum == 3 or linenum == 6:
                                sr = linenum//3
                                sc = linenum%3
                                locs = [(i,j) for i in range(sr,sr+2) for j in range(sc, sc+2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                                locs = [(i,j) for i in range(sr-1,sr+1) for j in range(sc, sc+2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                            elif linenum == 1 or linenum == 2:
                                sr = 0
                                sc = linenum
                                locs = [(i,j) for i in range(sr,sr+2) for j in range(sc, sc+2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                                locs = [(i,j) for i in range(sr,sr+2) for j in range(sc-1, sc+1) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                            else:
                                sr = linenum//3
                                sc = linenum%3
                                locs = [(i,j) for i in range(sr-1,sr+1) for j in range(sc-1, sc+1) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                                locs = [(i,j) for i in range(sr-1,sr+1) for j in range(sc, sc+2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                                locs = [(i,j) for i in range(sc, sc+2) for j in range(sr-1,sr+1) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                                locs = [(i,j) for i in range(sc, sc+2) for j in range(sc, sc+2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)

        if 1 in np.ravel([item for sublist in win_conditions_eval for item in sublist]):
            self.update_eval_board_1(available_locs, selected_piece)

        return np.argmax(self.eval_board)


class AI:
    """
    Class of AI player. this class is instantiated with
    current state of the board and the list of available pieces.

    attributes:
    - board: the current state of the board
    - available_pieces: the list of available pieces
    - is_first: whether the AI is the first player
    """

    def __init__(self, board, available_pieces: list[PieceType], is_first: bool):
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
        self.submachine = Submachine(board, available_pieces)

    def select_piece(self) -> PieceType:
        """
        Select a piece from the list of available pieces.

        return:
        - piece: the selected piece
        """

        if self.turn <= 4:
            return self.submachine.select_piece()

        self.build()

        state = self.to_state(self.board, None)
        best_child = None
        best_score = -1
        for child in self.children[state]:
            score = (self.wins[child]) / (self.visits[child] + 1)
            if score > best_score:
                best_child = child
                best_score = score
        *_, pidx = best_child
        return self.to_piece(pidx)

    def place_piece(self, piece: PieceType) -> PositionType:
        """
        Place the piece on the board.

        arguments:
        - piece: the piece to place

        return:
        - position: the position to place the piece (row, column)
        """
        if self.turn <= 4:
            return self.submachine.place_piece(piece)

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
                    if score > best_score:
                        best_move = (r, c)
                        best_score = score
        return best_move

    def build(self):
        """
        Build the monte carlo tree.
        """
        started_at = time.time()
        cnt = 0
        while time.time() - started_at < TL[self.turn - 1]:
            for _ in range(1000):
                self.simulate()
            cnt += 1

    def make_children(self, state: tuple[int], board, pieces: list[PieceType],
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
        return expected + explored

    def check_win_condition(self, board) -> list[list[bool]]:
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

        parent_state = None
        # run until unvisited node
        state = self.to_state(board, self.to_index(selected_piece) if selected_piece else None)
        while pieces:
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
        for state in history:
            self.visits[state] += 1
            self.wins[state] += result

    def is_me(self, pieces: list[PieceType], selected_piece: PieceType | None) -> bool:
        """
        Returns whether the AI is the player.
        """
        return (len(pieces) + bool(selected_piece)) % 2 == self.is_first

    @staticmethod
    def to_state(board, selected_index: int | None) -> tuple:
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
    def check_win_by_move(board, move: tuple) -> bool:
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
    def check_win(board) -> bool:
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
    def get_group_win_condition(group) -> list[tuple[int, bool]]:
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
    def is_group_won(group) -> bool:
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
    def instance(board, available_pieces: list[PieceType]) -> AI:
        return AI(board, available_pieces, is_first)

    return instance

