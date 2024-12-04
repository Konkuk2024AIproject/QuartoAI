import numpy as np
import random
from itertools import product
import copy
# 필승전략 : 어떤특성이 겹치는거 3개, 그 반대특성이 겹치는거 3개를 동시에 상대가 만들도록 한다
# 아니면 내가 33 만들고 상대한테 그거랑 최대한 다른말 주기(폭탄돌리기)

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = copy.deepcopy(board) # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = copy.deepcopy(available_pieces) # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.eval_board = np.zeros((4, 4), dtype=int)
        self.eval_piece = np.zeros(len(self.pieces))
    
    def select_piece(self):
        # eval_piece에서 가장 높은 곳 return
        if(self.board.max()):
            self.evaluate_piece()
            while(1):
                best_piece = self.pieces[self.eval_piece.argmax()]
                if(best_piece in self.available_pieces): # best_piece가 available한 경우
                    # print("eval piece : \n", self.eval_piece)
                    return best_piece
                else: # best_piece가 available하지 않은 경우 그 piece를 eval_piece의 최솟값으로 변경
                    self.eval_piece[self.eval_piece.argmax()] -= 500
        else:
            return random.choice(self.available_pieces) #랜덤 선택
            

    def place_piece(self, selected_piece):
        # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        # Available locations to place the piece
        # 0~15의 값을 return하면 해당 index에 둔다
        best_pos = self.evaluate_position(selected_piece)
        # print(best_pos)
        pos = random.choice(best_pos)
        # print(pos)
        row = pos // 4
        col = pos % 4
        # print(f"({row},{col})")
        return (row, col)


    # 완성도 체크 함수 : (-1,0,-1,-1),(-1,2,-1,-1) 이렇게 return하는게 좋을 듯
    # -1은 안겹쳐서 신경 안써도 되는거, 앞 튜플은 겹치는 mbti(0,1) 인코딩한거, 뒤에 있는 튜플은 그 mbti가 나온 횟수
    # mbti가 나온 횟수는 -1(신경안써도 되는거)나 다 같은 숫자이다.
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
                else: #빈리스트가 아닐 경우 
                    sym.append(u_val[0])
                    quantity.append(count[0])
        return sym, quantity # 겹치는게 없으면 그 속성에 대한 quantity 내 원소는 0이다


    #멤버변수 리셋하는 함수
    def member_reset(self):
        self.row_sym = []
        self.row_eval = []
        self.col_sym = []
        self.col_eval = []
        self.cross_sym = []
        self.cross_eval = [] # 좌->우 대각선, 우->좌 대각선
        self.subgrid_sym = []
        self.subgrid_eval = [] # 2X2 grid


    def check_possibilities(self):
        # 이 함수에서 2이상이 나온 부분이 중요한 부분임
        self.member_reset()
        
        for col in range(4):
            sym, quantity = self.check_possibility([self.board[row][col] for row in range(4)])
            if(len(sym)!= 0 and len(quantity) != 0 ):
                self.col_sym.append(sym)
                self.col_eval.append(quantity) # 각 col의 특성개수를 센 길이 4의 리스트가 col개수만큼 들어있다
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
        
        # 이걸 실행하면 이제 각 멤버변수에 각 승리가능 필드에서
        # 눈여겨 봐야할 특성,갯수가 들어간다
    
    
    def evaluate_piece(self):
        self.check_possibilities()  # 필드 평가를 통해 가능한 승리 조건 분석
        max_vals = [max([max(reval) for reval in self.row_eval]), max([max(ceval) for ceval in self.col_eval]), max([max(creval) for creval in self.cross_eval]), max([max(seval) for seval in self.subgrid_eval])]

        # 추가로, 남은 말 중에 내가 이길 수 있도록 해주는 특성을 가지고 있지 않은 말을 먼저 선택하는게 좋다(아직 구현 안함)

        if(3 in np.ravel(np.array(self.row_eval))):
            # 내가 고를 때 같은게 3개 중첩되어있는게 있으면 그 attribute는 piece선택에서 최대한 배제해야함
            for sindex, row in enumerate(self.row_sym):
                for idx, sym in enumerate(row) :
                    if(sym != -1 and self.row_eval[sindex][idx] == 3):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece -= np.array(result_array)
        
        if(3 in np.ravel(np.array(self.col_eval))):
            # 내가 고를 때 같은게 3개 중첩되어있는게 있으면 그 attribute는 piece선택에서 최대한 배제해야함
            for sindex, col in enumerate(self.col_sym):
                for idx, sym in enumerate(col) :
                    if(sym != -1 and self.col_eval[sindex][idx] == 3):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece -= np.array(result_array)
        
        if(3 in np.ravel(np.array(self.cross_eval))):
            # 내가 고를 때 같은게 3개 중첩되어있는게 있으면 그 attribute는 piece선택에서 최대한 배제해야함
            for sindex, cross in enumerate(self.cross_sym):
                for idx, sym in enumerate(cross) :
                    if(sym != -1 and self.cross_eval[sindex][idx] == 3):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece -= np.array(result_array)

        if(3 in np.ravel(np.array(self.subgrid_eval))):
            # 내가 고를 때 같은게 3개 중첩되어있는게 있으면 그 attribute는 piece선택에서 최대한 배제해야함
            for sindex, subgrid in enumerate(self.subgrid_sym):
                for idx, sym in enumerate(subgrid):
                    if(sym != -1 and self.subgrid_eval[sindex][idx] == 3):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([80 if piece in valid_piece_set else 0 for piece in self.pieces])
                        self.eval_piece -= np.array(result_array)
        
        if(2 in np.ravel(np.array(self.row_eval))):
            # 내가 고를 때 같은게 2개 중첩되어 있는게 있으면 그 attribute가 포함된 piece를 선택하는게 유리함
            # 상대가 둬서 3을 만들면 좋으니까
            for sindex, row in enumerate(self.row_sym):
                for idx, sym in enumerate(row) :
                    if(sym != -1 and self.row_eval[sindex][idx] == 2):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])
                        # 그 attrubute를 가지고 있는 piece가 많이 남아있으면 나한테 더 좋은 piece임
                        weight = np.sum(result_array)
                        self.eval_piece += (weight*np.array(result_array))

        if(2 in np.ravel(np.array(self.col_eval))):
            # 내가 고를 때 같은게 2개 중첩되어 있는게 있으면 그 attribute가 포함된 piece를 선택하는게 유리함
            # 상대가 둬서 3을 만들면 좋으니까
            for sindex, col in enumerate(self.col_sym):
                for idx, sym in enumerate(col) :
                    if(sym != -1 and self.col_eval[sindex][idx] == 2):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])
                        # 그 attrubute를 가지고 있는 piece가 많이 남아있으면 나한테 더 좋은 piece임
                        weight = np.sum(result_array)
                        self.eval_piece += (weight*np.array(result_array))

        if(2 in np.ravel(np.array(self.cross_eval))):
            # 내가 고를 때 같은게 2개 중첩되어 있는게 있으면 그 attribute가 포함된 piece를 선택하는게 유리함
            # 상대가 둬서 3을 만들면 좋으니까
            for sindex, cross in enumerate(self.cross_sym):
                for idx, sym in enumerate(cross) :
                    if(sym != -1 and self.cross_eval[sindex][idx] == 2):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])
                        # 그 attrubute를 가지고 있는 piece가 많이 남아있으면 나한테 더 좋은 piece임
                        weight = np.sum(result_array)
                        self.eval_piece += (weight*np.array(result_array))

        if(2 in np.ravel(np.array(self.subgrid_eval))):
            # 내가 고를 때 같은게 2개 중첩되어 있는게 있으면 그 attribute가 포함된 piece를 선택하는게 유리함
            # 상대가 둬서 3을 만들면 좋으니까
            for sindex, subgrid in enumerate(self.subgrid_sym):
                for idx, sym in enumerate(subgrid) :
                    if(sym != -1 and self.subgrid_eval[sindex][idx] == 2):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([1 if piece in valid_piece_set else 0 for piece in self.pieces])
                        # 그 attrubute를 가지고 있는 piece가 많이 남아있으면 나한테 더 좋은 piece임
                        weight = np.sum(result_array)
                        self.eval_piece += (weight*np.array(result_array))
        
        # 상대 차례에 val이 2인곳을 여러군데 만들게 해야함
        if(1 in max_vals): # max_vals에 1인 곳이 존재하면(게임 초반)
            #그곳에 맞는 특성에 1 더함
            for sindex, row in enumerate(self.row_sym):
                for idx, sym in enumerate(row):
                    if(sym != -1 and self.row_eval[sindex][idx] == 1):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([2 if piece in valid_piece_set else 0 for piece in self.pieces])
                        # val이 1인곳을 2로 만드는 piece는 상대에게 줬을 때 board에 2인 val들이 많아지게 한다
                        self.eval_piece += np.array(result_array)
            for sindex, col in enumerate(self.col_sym):
                for idx, sym in enumerate(col):
                    if(sym != -1 and self.col_eval[sindex][idx] == 1):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([2 if piece in valid_piece_set else 0 for piece in self.pieces])
                        # val이 1인곳을 2로 만드는 piece는 상대에게 줬을 때 board에 2인 val들이 많아지게 한다
                        self.eval_piece += np.array(result_array)
            for sindex, cross in enumerate(self.cross_sym):
                for idx, sym in enumerate(cross):
                    if(sym != -1 and self.cross_eval[sindex][idx] == 1):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([2 if piece in valid_piece_set else 0 for piece in self.pieces])
                        # val이 1인곳을 2로 만드는 piece는 상대에게 줬을 때 board에 2인 val들이 많아지게 한다
                        self.eval_piece += np.array(result_array)
            for sindex, subgrid in enumerate(self.subgrid_sym):
                for idx, sym in enumerate(subgrid):
                    if(sym != -1 and self.subgrid_eval[sindex][idx] == 1):
                        valid_piece = [piece for piece in self.available_pieces if piece[idx] == sym]
                        valid_piece_set = set(valid_piece)  # valid_piece를 set으로 변환
                        result_array = np.array([2 if piece in valid_piece_set else 0 for piece in self.pieces])
                        # val이 1인곳을 2로 만드는 piece는 상대에게 줬을 때 board에 2인 val들이 많아지게 한다
                        self.eval_piece += np.array(result_array)

    # 해당 row, col의 eval 값, 대각선 2개의 eval값, subgrid 9개의 eval값 해서 총 13개 원소 return
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
    

    # 지정된 값(1,2,3)이 위치한 row, col, cross, grid의 인덱스를 전부 파악
    def check_position(self, win_conditions_eval, same_value):
        max_positions = []
        for cond_idx, win_condition in enumerate(win_conditions_eval):
            max_positions.append([(cond_idx, linenum, symidx) for linenum, line in enumerate(win_condition) for symidx, val in enumerate(line) if val == same_value])
        
        return max_positions
        # (row인지 col인지 cross인지 grid인지, 몇번째인지, max attribute의 index)
        # (0: row, 1: col, 2: cross, 3: grid)
        # max_positions에는 각 승리조건의 최댓값을 가지고 있는 위치와, 심볼을 알 수 있다.

    def update_eval_board_2(self, symidx, locs, selected_piece):
        for (r,c) in locs:
            self.board[r][c] = self.pieces.index(selected_piece) + 1
            # 내가 받은걸 넣었을 때 val값이 바뀌는지 체크해봐야함
            # 2였다가 3이 되면 남은 말 중에 4로 만들 수 있는게 있는지 확인
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece)) # 3이 되는 부분이 있는지 확인

            if recalculated_max_val == 3:
                #남은 말 중에 4로 만드는게 남았는지 체크
                for r_sym in self.row_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if(piece[symidx] == r_sym[symidx]):
                            piece_make_4s.append(piece)
                    self.eval_board[r][c] += (len(piece_make_4s)) # 4로 만드는 말이 많이 남았을 수록 좋은 수임(상대가 나한테 줄게 없으니까)
                for c_sym in self.col_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if(piece[symidx] == c_sym[symidx]):
                            piece_make_4s.append(piece)
                    self.eval_board[r][c] += (len(piece_make_4s)) # 4로 만드는 말이 많이 남았을 수록 좋은 수임
                for cr_sym in self.cross_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if(piece[symidx] == cr_sym[symidx]):
                            piece_make_4s.append(piece)
                    self.eval_board[r][c] += (len(piece_make_4s)) # 4로 만드는 말이 많이 남았을 수록 좋은 수임
                for sg_sym in self.subgrid_sym:
                    piece_make_4s = []
                    for piece in self.available_pieces:
                        if(piece[symidx] == sg_sym[symidx]):
                            piece_make_4s.append(piece)
                    self.eval_board[r][c] += (len(piece_make_4s)) # 4로 만드는 말이 많이 남았을 수록 좋은 수임
                
            else: # 말을 뒀는데 val의 최댓값이 똑같이 2임(2는 많이 만들수록 좋음)
                self.eval_board[r][c] += 3 # 나쁘지 않은 수임
                
            self.board[r][c] = 0 # 원상복구

    def update_eval_board_1(self, locs, selected_piece):
        for (r,c) in locs:
            self.board[r][c] = self.pieces.index(selected_piece) + 1
            # 내가 받은걸 넣었을 때 val값이 바뀌는지 체크해봐야함
            # 2를 최대한 많이 만들도록 한다.
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece)) # 2가 되는 부분이 있는지 확인

            if recalculated_max_val == 2:
                self.eval_board[r][c] += 2
            else: # 말을 뒀는데 val의 최댓값이 똑같이 1임
                self.eval_board[r][c] += 1 #나쁘지 않은 수임

            self.board[r][c] = 0 # 원상복구

    # 위험한 곳의 위치를 파악하고 내가 갖고 있는 piece가 어디 들어가면 좋은지 확인
    def evaluate_position(self, selected_piece):
        self.check_possibilities()
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        win_conditions_eval = [self.row_eval, self.col_eval, self.cross_eval, self.subgrid_eval]
        win_conditions_sym = [self.row_sym, self.col_sym, self.cross_sym, self.subgrid_sym]
        
        max_vals = []
        # **그냥 0이 아닌 부분에 뒀을 때 나한테 좋은지 안좋은지 eval_board에 업데이트
        # tree 형태가 아니라 eval_board를 이용해 전역적으로 값을 업데이트 해가며 최적의 값을 찾는다

        # 나에게 도움이 되는 칸은 +1, 도움이 안되는 칸은 -1한다
        # 마지막에 최댓값을 가지고 있는 칸을 return하면 됨
        for win_condition in win_conditions_eval:
            #row, col, cross, grid의 최댓값 찾기
            max_vals.append(max(np.ravel(np.array(win_condition)))) #승리조건들의 각 요소의 최댓값 중 최댓값을 찾는다
            #row, col, cross, grid의 최댓값이 순서대로 들어감
        
        for r in range(1,3):
            for c in range(1,3):
                self.eval_board[r][c] += 1
        # eval_board의 중앙 부분에 가중치를 둬서 점수가 같은 곳이 있다면 가운데 부분에 우선적으로 두도록 한다

        # -1을 return하면 eval_board 확인해서 가장 큰 곳에 둔다
        # 0~15의 값을 return하면 해당 index에 둔다

        # 내가 가지고 있는 말을 뒀을 때 이길 수 있는 곳이 있으면 바로 return
        for loc in available_locs:
            r, c = loc
            self.board[r][c] = self.pieces.index(selected_piece) + 1
            recalculated_max_val = max(self.calculate_evaluation_value(r, c, selected_piece))

            if recalculated_max_val == 4:
                return [4*r + c]
            self.board[r][c] = 0

        # val이 3인 곳이 존재할 때 (eval_board 업데이트)
        # -> 내가 val을 4로 만들 수 있는 조각이 있다면 val을 4로 하는 위치에 둔다
        # -> val을 4로 만들 수 있는 위치가 없는 경우라 eval_board 업데이트
        if(3 in np.ravel([item for sublist in win_conditions_eval for item in sublist])):
            max_positions = self.check_position(win_conditions_eval, 3) # 3인 곳을 찾아줌
            # print("max_positions : ", max_positions)
            for cond in max_positions:
                if(len(cond) != 0):
                    for max_position in cond:
                        cond_idx, linenum, symidx = max_position
                        if(cond_idx == 0):
                            locs = [(r,c) for (r,c) in available_locs if r==linenum]
                        elif(cond_idx == 1):
                            locs = [(r,c) for (r,c) in available_locs if c==linenum]
                        elif(cond_idx == 2):
                            locs = [(r,c) for (r,c) in available_locs if(r==c or r+c==3)] # 대각선
                        else:
                            r = linenum//3
                            c = linenum%3
                            locs = [(r,c),(r,c+1),(r+1,c),(r+1,c+1)]
                        # print("locs: ", locs)
                        for (r,c) in locs:
                            self.board[r][c] = self.pieces.index(selected_piece) + 1
                            # 먼저, 현재 어떤 attribute가 3개 겹치는 승리 조건을 무효화 할 수 있는 곳이 있다면 가중치를 둔다
                            # 많은 attribute 최대한 겹치지 않게 할 수 있는 곳일 수록 (승리조건 성립을 무효화 할 수 있을 수록) 그곳에 놓는것이 유리
                            if(selected_piece[symidx] != win_conditions_sym[cond_idx][linenum][symidx]):
                                self.eval_board[r][c] += 10

                            # 내가 받은걸 넣어도 val값이 안바뀔거다(for문을 통과했으니까)
                            # 해당부분에 말을 뒀을 때 다른 부분이 3이 되는지 확인한다
                            # => pos_list에서 3인 곳이 있는지, 있다면 두 특성이 다른지 확인 두 특성이 다르다면 -80한다
                            # 이런식으로 계속...

                            pos_list = self.check_position(win_conditions_eval, 3) # board가 변경되었을 때 다시 3인 곳을 찾아줌
                            # print("find 3 : ", pos_list)
                            for pos_cond in pos_list:
                                if(len(pos_cond)!=0):
                                    for pos in pos_cond:
                                        new_cond_idx, new_linenum, new_symidx = pos
                                        if(symidx == new_symidx): # 같은 symbol이 겹치는지 확인
                                            # 해당 심볼이 다른지 확인(원래 3이었던 값의 symbol와 새로 3이된 symbol)
                                            sym_to_check = win_conditions_sym[cond_idx][linenum][symidx]
                                            if(win_conditions_sym[new_cond_idx][new_linenum][new_symidx] != sym_to_check):
                                                # 다르다면 eval_board[r][c]에 -80한다(상대가 무조건 이길 수 있는 수이기 때문에)
                                                self.eval_board[r][c] -= 80
                                            else: # 같다면 available piece 중에 해당 심볼을 무효화 할 수 있는게 몇개 남았는지 체크
                                                possible_pieces = [piece for piece in self.available_pieces if piece[new_symidx] != sym_to_check]
                                                self.eval_board[r][c] += len(possible_pieces)

                            self.board[r][c] = 0

        
        # 이건 max값이 2인 경우니까 내가 두고 다음 차례에 4가 되지 않게 하는 말이 남았으면
        # 그 위치의 eval_board에 1을 더하고, 아니면 다음걸 평가한다.
        # 가장 많은 위치를 return => place_piece에서 중앙부분에 더 좋은게 있는지 한번 찾는다
        
        # (row인지 col인지 cross인지 grid인지, 몇번째인지, max attribute의 index)
        # (0: row, 1: col, 2: cross, 3: grid)
        if(2 in np.ravel([item for sublist in win_conditions_eval for item in sublist])):
            max_positions = self.check_position(win_conditions_eval, 2) # 2인 곳을 찾아줌
            
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
            # 2를 최대한 많이 만들면 좋다
            self.update_eval_board_1(available_locs, selected_piece)
            
        # eval_board update후에는 가장 큰 값의 index를 return
        best_places = []

        if(len(available_locs) > 7):
            while(len(best_places) < 2):
                max_place = np.argmax(self.eval_board)
                row = max_place//4
                col = max_place%4
                if (row,col) in available_locs:
                    best_places.append(max_place)
                    self.eval_board[row][col] -= 500
                else:
                    self.eval_board[row][col] -= 500
        else:
            while(1):
                if(len(best_places)==1): break
                max_place = np.argmax(self.eval_board)
                row = max_place//4
                col = max_place%4
                if (row,col) in available_locs:
                    best_places.append(max_place)
                    self.eval_board[row][col] -= 500
                else:
                    self.eval_board[row][col] -= 500
        # print(best_places)
        return best_places # 1차원 배열 형식으로 변경해 최상위 3개 index 반환