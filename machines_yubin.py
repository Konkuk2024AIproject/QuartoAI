import numpy as np
from itertools import product

import time

class P2:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]  # All 16 pieces
        self.board = board  # 현재 보드 상태
        self.available_pieces = available_pieces    # 사용가능한 말 리스트
        # 히스토리 테이블 : 탐색 중복을 방지하기 위해 중간 상태를 저장하는 해시 테이블 -> 성능 개선
        self.history_table = {}  

    # 1. 현재 보드 상태 평가
    # 평가 함수 : 현재의 보드 상태 점수화해서 평가
    def evaluate_board(self, board):
        score = 0
        for row in range(4):    #모든 보드를 탐색
            for col in range(4):
                if board[row][col] != 0:  # 놓여진 말만 평가한다
                    piece_idx = board[row][col] - 1 # 보드의 해당 칸에 말의 고유 번호를 지정 (piece의 리스트에서 말의 인덱스와 연결 시킴)
                    piece = self.pieces[piece_idx] # 말의 정보 지정 및 가져옴
                    # 가져온 말의 정보가 승리에 가까운 줄을 만들 수 있으면 높은 점수 부여
                    # 줄 별로 점수를 매겨서 승리에 가까운 줄에 말 놓게 할 예정
                    score += self.evaluate_lines(board, row, col, piece)
                    score += self.evaluate_2x2_grids(board, row, col, piece)
        # 현재 보드 상태에서 승리 가능성 점수 추가
        score += self.evaluate_potential_wins(board)    
        return score

    # 가로 세로 대각선 라인을 평가
    # - 라인에 동일한 특성이 몇 개인지 
    # - 동일 특성일 수록 점수 높아짐 
    def evaluate_lines(self, board, row, col, piece):
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 가로, 세로, 대각선
        for dr, dc in directions:   # 네 방향을 차례대로 확인
            line = []
            for i in range(4):
                r = row + dr * i
                c = col + dc * i
                if 0 <= r < 4 and 0 <= c < 4: # 유효한 좌표에 있다면 해당 칸을 line리스트에 추가한다
                    line.append(board[r][c])
            if len(line) == 4: # 추가된 리스트의 길이가 4라면 평가를 시작한다
                characteristics = [self.pieces[idx - 1] for idx in line if idx != 0]    # line에 포함된 말의 정보를 확인
                if characteristics:
                    for i in range(4):  # 각 특성을 개별적으로 체크 4가지
                        # 해당 특성의 값을 추출하여 중복 여부 확인
                        trait_values = [char[i] for char in characteristics]
                        if len(set(trait_values)) == 1:  # 해당 특성이 같으면
                            score += len(trait_values) ** 2  # 특성의 개수에 비례한 점수 부여 (제곱은 더 높은 점수를 부여하기 위한 장치)
        return score    # 말이 포함된 가로 세로 대각선에서 획득한 총 점수를 반환
    
    # 2x2 그리드를 평가
    def evaluate_2x2_grids(self, board, row, col, piece):
        score = 0
        # 현재 위치 기준으로 포함되는 모든 2x2를 검사
        for dr in (0, -1):
            for dc in (0, -1):
                r, c = row + dr, col + dc
                if 0 <= r <= 2 and 0 <= c <= 2:  # 2x2가 유효한 범위 내에 있어야 함
                    grid = [
                        board[r][c], board[r][c + 1],
                        board[r + 1][c], board[r + 1][c + 1]
                    ]
                    if 0 not in grid:  # 2x2가 모두 차 있으면 평가
                        characteristics = [self.pieces[idx - 1] for idx in grid]
                        for i in range(4):  # 각 특성에 대해 평가
                            count = sum(char[i] == characteristics[0][i] for char in characteristics)  # 동일 특성 개수
                            if count > 1:  # 동일한 특성이 2개 이상이면 가중치 부여
                                score += count ** 2  # 제곱 방식으로 가중치 부여
        return score    # 말이 포함된 2x2그리드에서 획득한 총 점수 반환
    
    def evaluate_potential_wins(self, board):
    # 승리 가능성(특성 일치)을 점수로 환산
        score = 0
        for row in range(4):
            for col in range(4):
                if board[row][col] == 0:  # 빈 칸
                    board[row][col] = 1  # 임시로 말 배치
                    if self.check_win():
                        score += 50  # 임박한 승리에 높은 점수
                    board[row][col] = 0  # 복구
        return score

    # 보드의 상태를 해시로 변환해서 테이블에서 사용
    def state_hash(self, board):
        return hash(tuple(map(tuple, board)))

    def dynamic_depth(self, board):
        remaining_pieces = len(self.available_pieces)
        if remaining_pieces > 14:  # 초반: 탐색 깊이를 낮게
            return 4
        elif remaining_pieces > 10:  # 중반: 탐색 깊이를 중간 수준으로
            return 8
        else:  # 후반: 깊이 탐색 강화
            return 13

    # 2. Minimax 알파-베타 가지치기
    def minmax_ab(self, board, depth, is_maximizing, selected_piece, alpha, beta):
         # 깊이 조절
        if depth is None:
            depth = self.dynamic_depth(board)  # 현재 상태에서 적절한 깊이를 계산
        
        board_hash = self.state_hash(board) # 과거에 탐색했던 상태에 대해 중복 계산을 방지
        if board_hash in self.history_table:
            return self.history_table[board_hash]

        # 종료 조건
        if self.check_win():
            result = 1 if is_maximizing else -1 # 승리 조건에 따라 점수 반환
            self.history_table[board_hash] = result
            return result
        if self.is_board_full() or depth == 0:
            result = self.evaluate_board(board)
            self.history_table[board_hash] = result
            return result

        # 유효한 움직임을 정렬 => 알파-베타 가지치기의 효율을 높이기 위함
        moves = self.get_valid_moves(board) # 가능한 모든 움직임 
        # 각 움직임을 평가해서 정렬한다 
        # - 최대화 플레이어 : 내림차순
        # - 최소화 플레이어 : 오름차순
        moves.sort(key=lambda move: self.evaluate_move(board, move, selected_piece), reverse=is_maximizing)

        # 본격적 minmax알고리즘 구현
        best_value = -1e9 if is_maximizing else 1e9
        for row, col in moves:
            board[row][col] = self.pieces.index(selected_piece) + 1
            if is_maximizing:   # 재귀함수 사용핵서 수행
                eval = self.minmax_ab(board, depth - 1, False, selected_piece, alpha, beta)
                best_value = max(best_value, eval)
                alpha = max(alpha, eval)
            else:
                eval = self.minmax_ab(board, depth - 1, True, selected_piece, alpha, beta)
                best_value = min(best_value, eval)
                beta = min(beta, eval)
            board[row][col] = 0
            if beta <= alpha:   # 가지치기
                break
            
        # 계산된 최적의 값을 테이블에 저장하고 반환
        self.history_table[board_hash] = best_value
        return best_value
    
    # 움직임을 평가하는 함수 : 주어진 움직임의 점수를 계산 -> 플레이어가 말을 두는 즉각적 행동 평가
    def evaluate_move(self, board, move, piece):
        row, col = move
        board[row][col] = self.pieces.index(piece) + 1
        score = self.evaluate_board(board)
        board[row][col] = 0
        return score

    def get_valid_moves(self, board): # 유효한 움직임 반환
        return [(row, col) for row, col in product(range(4), repeat=2) if board[row][col] == 0]


    # 3. 실제 플레이어 행동
    def select_piece(self):
        return min(self.available_pieces, key=lambda piece: self.minmax_ab(self.board, None, False, piece, -1e9, 1e9))

    def place_piece(self, selected_piece):
        return max(self.get_valid_moves(self.board), key=lambda move: self.minmax_ab(self.board, None, True, selected_piece, -1e9, 1e9))


    # 4. 승리조건
    def check_win(self):
        for line in self.get_all_lines():
            if self.check_line(line):
                return True
        return self.check_2x2_subgrid_win()

    def get_all_lines(self):
        lines = []
        for i in range(4):
            lines.append([self.board[i][j] for j in range(4)])
            lines.append([self.board[j][i] for j in range(4)])
        lines.append([self.board[i][i] for i in range(4)])
        lines.append([self.board[i][3-i] for i in range(4)])
        return lines

    def check_line(self, line):
        if 0 in line:
            return False
        characteristics = np.array([self.pieces[piece_idx - 1] for piece_idx in line])
        return any(len(set(characteristics[:, i])) == 1 for i in range(4))

    def check_2x2_subgrid_win(self):
        for r, c in product(range(3), repeat=2):
            subgrid = [self.board[r+i][c+j] for i, j in product(range(2), repeat=2)]
            if 0 not in subgrid:
                characteristics = [self.pieces[idx - 1] for idx in subgrid]
                if any(len(set(char[i] for char in characteristics)) == 1 for i in range(4)):
                    return True
        return False

    def is_board_full(self):
        return all(all(cell != 0 for cell in row) for row in self.board)