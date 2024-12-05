import numpy as np
import random
from itertools import product
import copy

class Submachine():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = copy.deepcopy(board) # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.eval_board = np.zeros((4, 4), dtype=int)
        self.eval_piece = np.zeros(len(self.pieces))
    
    def select_piece(self):
        if(self.board.max()):
            self.evaluate_piece()
            while(1):
                best_piece = self.pieces[self.eval_piece.argmax()]
                if(best_piece in self.available_pieces):
                    return best_piece
                else:
                    self.eval_piece[self.eval_piece.argmax()] -= 500
        else:
            return random.choice(self.available_pieces)
            

    def place_piece(self, selected_piece):
        # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        # Available locations to place the piece
        if(self.board.max()):
            pos = self.evaluate_position(selected_piece)
            if(pos != 16) :
                while(1):
                    row = pos // 4
                    col = pos % 4
                    if((row,col) in available_locs):
                        return (row, col)
                    else:
                        self.eval_board[row][col] -= 500
                        pos = np.argmax(self.eval_board)
            else :
                return random.choice(available_locs)
        else:
            return (1,1)

    def check_possibility(self, line):
        line = [grid for grid in line if grid != 0]
        characteristics = np.array([self.pieces[idx - 1] for idx in line])
        sym, quantity = [],[]
        if(len(characteristics) == 0):
            for i in range(4):
                sym.append(-1)
                quantity.append(0)
        else :
            for char in characteristics.T:
                u_val, count = np.unique(char, return_counts=True)
                if(len(u_val) == 2):
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
            if(len(sym)!= 0 and len(quantity) != 0 ):
                self.col_sym.append(sym)
                self.col_eval.append(quantity)
            else:
                self.col_sym.append([-1,-1,-1,-1])
                self.col_eval.append([0,0,0,0])
    
        for row in range(4):
            sym, quantity = self.check_possibility([self.board[row][col] for col in range(4)])
            if(len(sym)!= 0 and len(quantity) != 0 ):
                self.row_sym.append(sym)
                self.row_eval.append(quantity)
            else:
                self.row_sym.append([-1,-1,-1,-1])
                self.row_eval.append([0,0,0,0])

        sym, quantity = self.check_possibility([self.board[i][i] for i in range(4)])
        if(len(sym)!= 0 and len(quantity) != 0 ):
            self.cross_sym.append(sym)
            self.cross_eval.append(quantity)
        else:
            self.cross_sym.append([-1,-1,-1,-1])
            self.cross_eval.append([0,0,0,0])
        sym, quantity = self.check_possibility([self.board[i][3 - i] for i in range(4)])
        if(len(sym)!= 0 and len(quantity) != 0 ):
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
                if(len(sym)!= 0 and len(quantity) != 0 ):
                    self.subgrid_sym.append(sym)
                    self.subgrid_eval.append(quantity)
                else:
                    self.subgrid_sym.append([-1,-1,-1,-1])
                    self.subgrid_eval.append([0,0,0,0])
        
    
    
    def evaluate_piece(self):
        self.check_possibilities()  
        max_vals = [max([max(reval) for reval in self.row_eval]), max([max(ceval) for ceval in self.col_eval]), max([max(creval) for creval in self.cross_eval]), max([max(seval) for seval in self.subgrid_eval])]

        if(3 in np.ravel(np.array(self.row_eval))):
            for sindex, row in enumerate(self.row_sym):
                for idx, sym in enumerate(row) :
                    if(sym != -1 and self.row_eval[sindex][idx] == 3):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)
                        result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece -= np.array(result_array)
        
        if(3 in np.ravel(np.array(self.col_eval))):
            for sindex, col in enumerate(self.col_sym):
                for idx, sym in enumerate(col) :
                    if(sym != -1 and self.col_eval[sindex][idx] == 3):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece -= np.array(result_array)
        
        if(3 in np.ravel(np.array(self.cross_eval))):
            for sindex, cross in enumerate(self.cross_sym):
                for idx, sym in enumerate(cross) :
                    if(sym != -1 and self.cross_eval[sindex][idx] == 3):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece -= np.array(result_array)

        if(3 in np.ravel(np.array(self.subgrid_eval))):
            for sindex, subgrid in enumerate(self.subgrid_sym):
                for idx, sym in enumerate(subgrid):
                    if(sym != -1 and self.subgrid_eval[sindex][idx] == 3):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece -= np.array(result_array)
        
        if(2 in np.ravel(np.array(self.row_eval))):
            for sindex, row in enumerate(self.row_sym):
                for idx, sym in enumerate(row) :
                    if(sym != -1 and self.row_eval[sindex][idx] == 2):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])
                        weight = np.sum(result_array)
                        self.eval_piece += (weight*np.array(result_array))

        if(2 in np.ravel(np.array(self.col_eval))):
            for sindex, col in enumerate(self.col_sym):
                for idx, sym in enumerate(col) :
                    if(sym != -1 and self.col_eval[sindex][idx] == 2):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])
                        weight = np.sum(result_array)
                        self.eval_piece += (weight*np.array(result_array))

        if(2 in np.ravel(np.array(self.cross_eval))):
            for sindex, cross in enumerate(self.cross_sym):
                for idx, sym in enumerate(cross) :
                    if(sym != -1 and self.cross_eval[sindex][idx] == 2):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])
                        weight = np.sum(result_array)
                        self.eval_piece += (weight*np.array(result_array))

        if(2 in np.ravel(np.array(self.subgrid_eval))):
            for sindex, subgrid in enumerate(self.subgrid_sym):
                for idx, sym in enumerate(subgrid) :
                    if(sym != -1 and self.subgrid_eval[sindex][idx] == 2):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])

                        weight = np.sum(result_array)
                        self.eval_piece += (weight*np.array(result_array))

        if(1 in max_vals):
            for sindex, row in enumerate(self.row_sym):
                for idx, sym in enumerate(row):
                    if(sym != -1 and self.row_eval[sindex][idx] == 1):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([2 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece += np.array(result_array)
            for sindex, col in enumerate(self.col_sym):
                for idx, sym in enumerate(col):
                    if(sym != -1 and self.col_eval[sindex][idx] == 1):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([2 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece += np.array(result_array)
            for sindex, cross in enumerate(self.cross_sym):
                for idx, sym in enumerate(cross):
                    if(sym != -1 and self.cross_eval[sindex][idx] == 1):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([2 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece += np.array(result_array)
            for sindex, subgrid in enumerate(self.subgrid_sym):
                for idx, sym in enumerate(subgrid):
                    if(sym != -1 and self.subgrid_eval[sindex][idx] == 1):
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
        return pos_list
    
    def check_position(self, win_conditions_eval, same_value):
        max_positions = []
        for cond_idx, win_condition in enumerate(win_conditions_eval):
            max_positions.append([(cond_idx, linenum, symidx) for linenum, line in enumerate(win_condition) for symidx, val in enumerate(line) if val == same_value])
        return max_positions


    def update_eval_board_2(self, symidx, locs, selected_piece):
        for (r,c) in locs:
            self.board[r][c] = self.pieces.index(selected_piece) + 1
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece)) # 3이 되는 부분이 있는지 확인
            if recalculated_max_val == 3:
                for r_sym in self.row_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if(piece[symidx] == r_sym[symidx]):
                            piece_make_4s.append(piece)
                            self.eval_board[r][c] -= (len(piece_make_4s))
                for c_sym in self.col_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if(piece[symidx] == c_sym[symidx]):
                            piece_make_4s.append(piece)
                            self.eval_board[r][c] -= (len(piece_make_4s))
                for cr_sym in self.cross_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if(piece[symidx] == cr_sym[symidx]):
                            piece_make_4s.append(piece)
                            self.eval_board[r][c] -= (len(piece_make_4s))
                for sg_sym in self.subgrid_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if(piece[symidx] == sg_sym[symidx]):
                            piece_make_4s.append(piece)
                            self.eval_board[r][c] -= (len(piece_make_4s))
            else: 
                self.eval_board[r][c] += 3 
            self.board[r][c] = 0 

    def update_eval_board_1(self, locs, selected_piece):
        for (r,c) in locs:
            self.board[r][c] = self.pieces.index(selected_piece) + 1
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece))
            if recalculated_max_val == 2:
                self.eval_board[r][c] += 1
            else:
                self.eval_board[r][c] += 1

            self.board[r][c] = 0 # 원상복구

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
        for loc in available_locs:
            r, c = loc
            self.board[r][c] = self.pieces.index(selected_piece) + 1
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece))

            if recalculated_max_val == 4:
                return (4*r + c)
            else : 
                self.board[r][c] = 0

        if(3 in np.ravel([item for sublist in win_conditions_eval for item in sublist])):
            max_positions = self.check_position(win_conditions_eval, 3) # 3인 곳을 찾아줌
            for cond in max_positions:
                if(len(cond)!= 0):
                    for max_position in cond:
                        cond_idx, linenum, symidx = max_position
                        locs = [(r,c) for (r,c) in available_locs if r==linenum]
                        for (r,c) in locs:
                            self.board[r][c] = self.pieces.index(selected_piece) + 1
                            pos_list = self.check_position(win_conditions_eval, 3) # board가 변경되었을 때 다시 3인 곳을 찾아줌
                            for pos_cond in pos_list:
                                if(len(pos_cond)!=0):
                                    for pos in pos_cond:
                                        new_cond_idx, new_linenum, new_symidx = pos
                                        if(symidx == new_symidx):
                                            sym_to_check = win_conditions_sym[cond_idx][linenum][symidx]
                                            if(win_conditions_sym[new_cond_idx][new_linenum][new_symidx] != sym_to_check):
                                                self.eval_board[r][c] -= 80
                                            else:
                                                possible_pieces = [piece for piece in self.available_pieces if piece[new_symidx] != sym_to_check]
                                                self.eval_board[r][c] += len(possible_pieces)
                            self.board[r][c] = 0
        if(2 in np.ravel([item for sublist in win_conditions_eval for item in sublist])):
            max_positions = self.check_position(win_conditions_eval, 2)
            
            for cond in max_positions:
                if(len(cond)!= 0):
                    for max_position in cond:
                        cond_idx, linenum, symidx = max_position
                        if(cond_idx == 0): # row
                            locs = [(r,c) for (r,c) in available_locs if (r==linenum and self.board[r][c] == 0)]
                            self.update_eval_board_2(symidx, locs, selected_piece)

                        elif(cond_idx == 1): # col
                            locs = [(r,c) for (r,c) in available_locs if (c==linenum and self.board[r][c] == 0)]
                            self.update_eval_board_2(symidx, locs, selected_piece)
                        
                        elif(cond_idx == 2): # cross
                            locs = [(i,i) for i in range(4) if (i==linenum and self.board[r][c] == 0)]
                            self.update_eval_board_2(symidx, locs, selected_piece)
                            locs = [(i,3-i) for i in range(4) if (i==linenum and self.board[r][c] == 0)]
                            self.update_eval_board_2(symidx, locs, selected_piece)
                        
                        else: # subgrids
                            if(linenum == 0):
                                locs = [(i,j) for i in range(2) for j in range(2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                            elif(linenum == 3 or linenum == 6):
                                sr = linenum//3
                                sc = linenum%3
                                locs = [(i,j) for i in range(sr,sr+2) for j in range(sc, sc+2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                                locs = [(i,j) for i in range(sr-1,sr+1) for j in range(sc, sc+2) if(self.board[i][j] == 0)]
                                self.update_eval_board_2(symidx, locs, selected_piece)
                            elif(linenum == 1 or linenum == 2):
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

        if(1 in np.ravel([item for sublist in win_conditions_eval for item in sublist])):
            self.update_eval_board_1(available_locs, selected_piece)

        return np.argmax(self.eval_board)