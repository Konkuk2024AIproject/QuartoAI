import numpy as np
import random
from itertools import product

import time

class P1():
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board # Include piece indices. 0:empty / 1~16:piece
        self.available_pieces = available_pieces # Currently available pieces in a tuple type (e.g. (1, 0, 1, 0))
        self.eval_board = np.zeros((4, 4), dtype=int)
    
    def select_piece(self):
        # Make your own algorithm here
        # eval_board에서 가장 높은 곳에 넣었을때 어떤 조건도 만족하지 못하는걸 선택

        if(self.board.max()):
            #row = self.evaluate_position() // 4
            #col = self.evaluate_position() % 4
            return self.evaluate_piece()
        # 첫 선택이면 그냥 무조건 ENFJ를 준다    
        else:
            return (1,0,1,1) #ENFJ
            

    def place_piece(self, selected_piece):
        # selected_piece: The selected piece that you have to place on the board (e.g. (1, 0, 1, 0)).
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        # Available locations to place the piece
        # 첫 수가 아닌 경우
        if(self.board.max()):
            pos = self.evaluate_position(selected_piece)
            row = pos // 4
            col = pos % 4
            if(pos != 16) :
                    # for r in range(1,3):
                    #     for c in range(1,3):
                    #         if(self.eval_board[r][c]==value):
                    #             row = r
                    #             col = c
                    #             break
                            # if(self.eval_board[row][col] != 0 and self.eval_board[row][col] <= max(value)):
                            #     return (r,c) # 중앙에 빈자리가 있으면 먼저 선점
                return (row, col)
            else :
                return random.choice(available_locs)
            

        else: # 내가 첫수라면 무조건 1,1에 둔다
            return (1,1)


    # 완성도 체크 함수 : (-1,0,-1,-1),max값 이렇게 return하는게 좋을 듯
    # -1은 안겹쳐서 신경 안써도 되는거, 앞 튜플은 겹치는 mbti(0,1) 인코딩한거, 뒤에 있는 튜플은 그 mbti가 나온 횟수
    # mbti가 나온 횟수는 -1(신경안써도 되는거)나 다 같은 숫자이다.
    def check_possibility(self, line):
        line = [grid for grid in line if grid != 0]
        characteristics = np.array([self.pieces[idx - 1] for idx in line])
        sym, quantity = [],[0]
        for char in characteristics.T:
            u_val, count = np.unique(char, return_counts=True)
            if(len(u_val) == 2):
                sym.append(-1)
                quantity.append(-1)
            else: 
                sym.append(u_val[0])
                quantity.append(count[0])
            
        return(sym, max(quantity)) # 겹치는게 없으면 quantity는 0이다


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
        # 이 함수에서 2이상이 나온 부분을 우선 봐서 
        # 내가 선택하는 순서면 거기랑 안맞는걸 주고 내가 놓는 순서면
        # 3인 곳을 찾아서 놓는다
        self.member_reset()
        
        for col in range(4):
            sym, quantity = self.check_possibility([self.board[row][col] for row in range(4)])
            self.col_sym.append(sym)
            self.col_eval.append(quantity)
    
        for row in range(4):
            sym, quantity = self.check_possibility([self.board[row][col] for col in range(4)])
            self.row_sym.append(sym)
            self.row_eval.append(quantity)

        sym, quantity = self.check_possibility([self.board[i][i] for i in range(4)])
        self.cross_sym.append(sym)
        self.cross_eval.append(quantity)
        sym, quantity = self.check_possibility([self.board[i][3 - i] for i in range(4)])
        self.cross_sym.append(sym)
        self.cross_eval.append(quantity)

        # Check 2x2 sub-grids
        for r in range(3):
            for c in range(3):
                subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                sym, quantity = self.check_possibility(subgrid)
                self.subgrid_sym.append(sym)
                self.subgrid_eval.append(quantity)
        
        # 이걸 실행하면 이제 각 멤버변수에 각 승리가능 필드에서
        # 눈여겨 봐야할 특성,갯수가 들어간다
                
                
    
    def evaluate_piece(self):
        self.check_possibilities()  # 필드 평가를 통해 가능한 승리 조건 분석
    
        # 빈 위치 찾기
        available_locs = [
            (row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0
        ]
        
        # 각 남은 조각에 대해 승리 가능성 평가
        eval_results = []
        for piece in self.available_pieces:
            score = 0
            
            # 각 빈 위치에 해당 조각을 놓았을 때 평가
            for loc in available_locs:
                row, col = loc
                self.board[row][col] = self.available_pieces.index(piece) + 1  # 임시로 조각 배치
                score += self.eval_board[row][col]
                self.board[row][col] = 0  # 원상 복구
            
            eval_results.append((piece, score))
        
        # 최고 점수를 가진 조각 탐색
        best_piece, max_score = max(eval_results, key=lambda x: x[1])
        
        # 선택 기준에 따라 반환
        if max_score >= 2:
            # max_score가 2 이상인 경우, 상대가 가장 두고 싶어하지 않을 조각 반환
            selected_piece = min(eval_results, key=lambda x: x[1])[0]
        else:
            # max_score가 2 미만인 경우, 적당한 조각 반환
            selected_piece = best_piece
        
        return selected_piece         


    # 위험한 곳의 위치를 파악하고 내가 갖고 있는 piece가 어디 들어가면 좋은지 확인
    def evaluate_position(self, selected_piece):
        self.check_possibilities()
        available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col]==0]
        max_row = max(self.row_eval) # 0,1,2,3
        max_col = max(self.col_eval) # 0,1,2,3
        max_cross = max(self.cross_eval) # 0, 1
        max_subgrid = max(self.subgrid_eval) # 0~8까지
        max_vals = [max_row, max_col, max_cross, max_subgrid]
        row_most_pos =  np.where(np.array(self.row_eval) == max_row)[0].tolist()
        col_most_pos =  np.where(np.array(self.col_eval) == max_col)[0].tolist()
        # cross_most_pos = self.cross_eval.index(max_cross)
        subgrid_most_pos = np.where(np.array(self.subgrid_eval) == max_subgrid)[0].tolist()

        self.eval_board = np.zeros((4, 4), dtype=int)
        # **그냥 0이 아닌 부분에 뒀을 때 나한테 좋은지 안좋은지 eval_board에 업데이트
        # tree 형태가 아니라 eval_board를 이용해 전역적으로 값을 업데이트 해가며 최적의 값을 찾는다

        # eval 값이 1이면 협력해도 됨, eval 값이 2이면 방해해야됨
        # 나에게 도움이 되는 칸은 +1, 도움이 안되는 칸은 -1한다
        # 마지막에 최댓값을 가지고 있는 칸을 return하면 됨
        
        if(max(max_vals) == 3):
            for loc in available_locs:
                r, c = loc
                pos_list = []
                self.board[r][c] = self.pieces.index(selected_piece) + 1
                _, count = self.check_possibility([self.board[r][col] for col in range(4)])
                pos_list.append(count)
                _, count = self.check_possibility([self.board[row][c] for row in range(4)])
                pos_list.append(count)
                _, count = self.check_possibility([self.board[i][i] for i in range(4)])
                pos_list.append(count)
                _, count = self.check_possibility([self.board[i][3-i] for i in range(4)])
                pos_list.append(count)

                if(r<=2 and c <=2):
                    subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                    _, count = self.check_possibility(subgrid)
                    pos_list.append(count)

                if max(pos_list) == 4:
                    return (4*r + c)
                else :
                    return 16 # 이길 수 있는 경우가 없는거임
            
        # row,col또는 row,cross 또는 col,cross에서 모두 eval 변수가 2인 순간 거의 게임 끝남
        # 이건 max값이 2인 경우니까 내가 두고 다음 차례에 4가 되지 않게 하는 말이 남았으면
        # 그 위치의 eval_board에 1을 더하고, 아니면 다음걸 평가한다.
        # elif(max(max_vals) == 2):
        # eval값들이 2이하인 경우 eval_board를 일단 업데이트 한 후
        # eval값들이 2가 아닌 경우(아마 게임 초반) 겹치는 부분이
        # 가장 많은 위치를 return => place_piece에서 중앙부분에 더 좋은게 있는지 한번 찾는다
        else: 
            if(max_row == 2):
                for row_pos in row_most_pos:
                    for i in range(4):
                        if(self.board[row_pos][i] == 0):
                            self.board[row_pos][i] = self.pieces.index(selected_piece)
                            _, pos_before = self.check_possibility([self.board[row_pos][col] for col in range(4)])
                            if(pos_before == 3):
                                # 4가 되지 않게 하는 말이 남아있어야됨
                                # 여기부터 상대 수 예측
                                for idx, piece in enumerate(self.available_pieces):
                                    colval = np.argmin(self.board[row_pos])
                                    self.board[row_pos][colval] = self.pieces.index(piece) + 1
                                    _, pos_after = self.check_possibility([self.board[row_pos][col] for col in range(4)])
                                    if (pos_after == 4): 
                                        # 남아있는 수 중에 승리조건을 만족하게 하는게 있으면
                                        # 내가 [row_most_pos][i]에 뒀을 때 진다.
                                        self.eval_board[row_pos][i] -= 1
                                    else: self.eval_board[row_pos][i] += 1
                                    self.board[row_pos][colval] = 0
                            self.board[row_pos][i] = 0

            if(max_col == 2):
                for col_pos in col_most_pos:
                    for i in range(4):
                        if(self.board[i][col_pos] == 0):
                            self.board[i][col_pos] = self.pieces.index(selected_piece)
                            _, pos_before = self.check_possibility([self.board[row][col_pos] for row in range(4)])
                            if(pos_before == 3):
                                # 4가 되지 않게 하는 말이 남아있어야됨
                                for idx, piece in enumerate(self.available_pieces):
                                    rowval = np.argmin(self.board.T[col_pos])
                                    self.board[rowval][col_pos] = self.pieces.index(piece) + 1
                                    _, pos_after = self.check_possibility([self.board[row][col_pos] for row in range(4)])
                                    if (pos_after == 4):
                                        self.eval_board[i][col_pos] -= 1
                                    else: self.eval_board[i][col_pos] += 1
                                    self.board[rowval][col_pos] = 0
                            self.board[i][max_col] = 0


            if(max_cross == 2):
                for diag in range(2):  # diag = 0: 좌->우 대각선, diag = 1: 우->좌 대각선
                    diagonal = ([self.board[i][i] for i in range(4)] if diag == 0
                                else [self.board[i][3 - i] for i in range(4)])
                    for i in range(4):
                        if diagonal[i] == 0:  # 빈 칸을 찾음
                            r, c = (i, i) if diag == 0 else (i, 3 - i)
                            self.board[r][c] = self.pieces.index(selected_piece) + 1
                            _, pos_before = self.check_possibility([self.board[d][d] if diag == 0 else self.board[d][3 - d] for d in range(4)])
                            if pos_before == 3:
                                # 4가 되지 않게 하는 말 탐색
                                for idx, piece in enumerate(self.available_pieces):
                                    self.board[r][c] = self.pieces.index(piece) + 1
                                    _, pos_after = self.check_possibility([self.board[d][d] if diag == 0 else self.board[d][3 - d] for d in range(4)])
                                    if pos_after == 4:
                                        self.eval_board[r][c] -= 1
                                    else:
                                        self.eval_board[r][c] += 1
                                    self.board[r][c] = 0
                            self.board[r][c] = 0 

            #서브그리드는 max인 곳에 추가하고 전체를 다 따져봐야한다
            if(max_subgrid == 2):
                for subgrid_pos in subgrid_most_pos:
                    r = subgrid_pos // 3
                    c = subgrid_pos % 3
                    subgrid = [self.board[r][c], self.board[r][c+1],
                            self.board[r+1][c], self.board[r+1][c+1]]
                    subgrid_indices = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
                    for idx, (r, c) in enumerate(subgrid_indices):
                            if self.board[r][c] == 0:  # 빈 칸을 찾음
                                self.board[r][c] = self.pieces.index(selected_piece) + 1
                                # 해당 서브그리드가 승리 조건에 가까운지 평가
                                _, pos_before = self.check_possibility([self.board[i][j] for i, j in subgrid_indices])
                                if pos_before == 3:
                                    # 4가 되지 않게 하는 말 탐색
                                    for piece in self.available_pieces:
                                        self.board[r][c] = self.pieces.index(piece) + 1
                                        _, pos_after = self.check_possibility([self.board[i][j] for i, j in subgrid_indices])
                                        if pos_after == 4:
                                            self.eval_board[r][c] -= 1
                                        else:
                                            self.eval_board[r][c] += 1
                                        self.board[r][c] = 0
                                self.board[r][c] = 0

            if(max(max_vals) < 2) :
                #selected piece랑 가장 유사한곳에 둔다
                max_val = max(max_vals)
                for loc in available_locs:
                    r,c = loc
                    pos_list = []
                    self.board[r][c] = self.pieces.index(selected_piece) + 1
                    _, count = self.check_possibility([self.board[r][col] for col in range(4)])
                    pos_list.append(count)
                    _, count = self.check_possibility([self.board[row][c] for row in range(4)])
                    pos_list.append(count)
                    _, count = self.check_possibility([self.board[i][i] for i in range(4)])
                    pos_list.append(count)
                    _, count = self.check_possibility([self.board[i][3-i] for i in range(4)])
                    pos_list.append(count)

                    subgrid = [self.board[r][c], self.board[r][c+1], self.board[r+1][c], self.board[r+1][c+1]]
                    _, count = self.check_possibility(subgrid)
                    pos_list.append(count)
                    self.board[r][c] = 0

                    if max(pos_list) > max_val:
                        return (4*r + c)
                    else :
                        return 16 # 랜덤하게 그냥 둔다

        return np.argmax(self.eval_board) # 1차원 배열 형식으로 변경해 가장 큰 index 반환
    