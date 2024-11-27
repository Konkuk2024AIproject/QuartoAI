import numpy as np
import random
from itertools import product
import time

class P1:
    def __init__(self, board, available_pieces):
        self.pieces = [(i, j, k, l) for i in range(2) for j in range(2) for k in range(2) for l in range(2)]
        self.board = board  # 0: empty / 1~16: piece index
        self.available_pieces = available_pieces  # 현재 사용할 수 있는 조각 목록
        self.total_execution_time = 0  # 총 실행 시간 누적
        self.start_time = None  # 함수 호출 시작 시간

    def _check_time_limit(self):
        """시간 제한을 초과했는지 확인."""
        return time.time() - self.start_time > 300

    def select_piece(self):
        """최적의 조각을 선택."""
        
        self.start_time = time.time()
        best_piece = None
        max_disrupt_score = -1

        for piece in self.available_pieces:
            if self._check_time_limit():
                break  # 시간 제한 초과 시 중단

            disrupt_score = sum(self._disrupt_opponent(piece, row, col)
                                for row in range(4) for col in range(4) if self.board[row][col] == 0)

            if disrupt_score > max_disrupt_score:
                max_disrupt_score = disrupt_score
                best_piece = piece

        # 시간이 초과되거나 조각 탐색이 완료되면 최적의 조각 반환
        time.sleep(0.5)
        return best_piece if best_piece else random.choice(self.available_pieces)

    def place_piece(self,selected_piece):
        """Minimax 알고리즘을 사용하여 최적의 위치를 선택."""
        self.start_time = time.time()  # 시작 시간 기록
        depth = 2  # 탐색 깊이 설정

        # Minimax를 사용하여 최적의 위치를 탐색
        _, best_move = self._minimax(self.board, self.available_pieces, depth, True)

        # Minimax가 None을 반환한 경우 기본값으로 임의의 빈 위치 반환
        if best_move is None:
            available_locs = [(row, col) for row, col in product(range(4), range(4)) if self.board[row][col] == 0]
            return random.choice(available_locs)

        return best_move


    def _disrupt_opponent(self, piece, row, col):
        """특정 위치에서 상대방의 승리 가능성을 방해하는 정도 평가."""
        disruption = 0
        
        # 세로 방향
        for r in range(4):
            if self.board[r][col] != 0:
                existing_piece = self.pieces[self.board[r][col] - 1]
                disruption += sum(1 for i in range(4) if piece[i] != existing_piece[i])
        
        # 가로 방향
        for c in range(4):
            if self.board[row][c] != 0:
                existing_piece = self.pieces[self.board[row][c] - 1]
                disruption += sum(1 for i in range(4) if piece[i] != existing_piece[i])
        
        # 왼쪽 위 → 오른쪽 아래 대각선
        if row == col:  # 위치가 대각선 상에 있을 경우
            for d in range(4):
                if self.board[d][d] != 0:
                    existing_piece = self.pieces[self.board[d][d] - 1]
                    disruption += sum(1 for i in range(4) if piece[i] != existing_piece[i])
        
        # 오른쪽 위 → 왼쪽 아래 대각선
        if row + col == 3:  # 위치가 역대각선 상에 있을 경우
            for d in range(4):
                if self.board[d][3 - d] != 0:
                    existing_piece = self.pieces[self.board[d][3 - d] - 1]
                    disruption += sum(1 for i in range(4) if piece[i] != existing_piece[i])
        
        # 2x2 네모 평가
        for r in range(max(0, row - 1), min(3, row + 1)):  # row를 중심으로 2x2 상단 행 찾기
            for c in range(max(0, col - 1), min(3, col + 1)):  # col을 중심으로 2x2 좌측 열 찾기
                if self.board[r][c] != 0 and self.board[r][c + 1] != 0 and \
                self.board[r + 1][c] != 0 and self.board[r + 1][c + 1] != 0:
                    # 2x2 네모에 포함된 조각 속성 가져오기
                    subgrid_pieces = [
                        self.pieces[self.board[r][c] - 1],
                        self.pieces[self.board[r][c + 1] - 1],
                        self.pieces[self.board[r + 1][c] - 1],
                        self.pieces[self.board[r + 1][c + 1] - 1],
                    ]
                    for i in range(4):  # 각 속성별로 확인
                        if all(p[i] == piece[i] for p in subgrid_pieces):  # 모든 조각의 i번째 속성이 같으면 방해
                            disruption += 1
        
        return disruption


    def _is_winning_move(self, row, col):
        """현재 위치에 놓으면 승리 조건을 만족하는지 확인."""
        lines_to_check = [
            [(row, j) for j in range(4)],  # 가로줄
            [(i, col) for i in range(4)],  # 세로줄
            [(i, i) for i in range(4)] if row == col else [],  # 대각선
            [(i, 3 - i) for i in range(4)] if row + col == 3 else []  # 역대각선
        ]
        
        # 가로, 세로, 대각선 확인
        for line in lines_to_check:
            for attribute_idx in range(4):  # 각 속성별로 확인
                if line and self._check_line(line, attribute_idx):
                    return True

        # 2x2 서브그리드 확인
        for r in range(max(0, row - 1), min(3, row + 1)):  # row를 중심으로 2x2 상단 행 찾기
            for c in range(max(0, col - 1), min(3, col + 1)):  # col을 중심으로 2x2 좌측 열 찾기
                if self._check_subgrid(r, c):  # 2x2 네모 확인
                    return True

        return False

    def _blocks_opponent_win(self, row, col):
        """특정 위치가 상대방의 승리를 막는지 확인."""
        return self._is_winning_move(row, col)

    def _check_line(self, line, attribute_idx):
        """주어진 라인의 속성이 모두 일치하는지 확인."""
        attributes = [self.pieces[self.board[r][c] - 1][attribute_idx] for r, c in line if self.board[r][c] != 0]
        return len(set(attributes)) == 1 and len(attributes) == 3  # 세 개가 일치하면 True

    def _check_subgrid(self, row, col):
        """2x2 네모의 속성이 모두 일치하는지 확인."""
        # 2x2 네모 내 위치들
        subgrid_coords = [(row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1)]
        
        # 서브그리드 내 조각의 속성을 가져옴
        subgrid_pieces = [self.pieces[self.board[r][c] - 1] for r, c in subgrid_coords if self.board[r][c] != 0]
        
        if len(subgrid_pieces) < 4:
            return False  # 서브그리드에 조각이 부족하면 승리 불가
        
        # 각 속성별로 확인
        for i in range(4):  # 속성 인덱스 (0~3)
            if len(set(p[i] for p in subgrid_pieces)) == 1:  # 모든 조각의 i번째 속성이 같으면 승리
                return True

        return False


    def _evaluate_position(self, row, col):
        """주어진 위치의 전략적 가치 평가."""
        disruption_score = 0
        for r in range(4):
            disruption_score += sum(1 for c in range(4) if self.board[r][c] == 0)
        for c in range(4):
            disruption_score += sum(1 for r in range(4) if self.board[r][c] == 0)
        return disruption_score

    def _minimax(self, board, available_pieces, depth, is_maximizing):
        """Minimax 알고리즘."""
        if depth == 0 or self._is_terminal_state(board):
            return self._evaluate_board_state(board), None

        best_move = None
        if is_maximizing:
            max_eval = float('-inf')
            for row, col in product(range(4), range(4)):
                if board[row][col] == 0:
                    # 승리 조건을 만족하면 즉시 반환
                    if self._is_winning_move(row, col):
                        return float('inf'), (row, col)

                    for piece in available_pieces:
                        new_board, new_pieces = self._simulate_move(board, available_pieces, row, col, piece)
                        eval, _ = self._minimax(new_board, new_pieces, depth - 1, False)
                        if eval > max_eval:
                            max_eval = eval
                            best_move = (row, col)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for row, col in product(range(4), range(4)):
                if board[row][col] == 0:
                    # 상대방이 승리할 가능성이 있으면 즉시 차단
                    if self._blocks_opponent_win(row, col):
                        return float('-inf'), (row, col)

                    for piece in available_pieces:
                        new_board, new_pieces = self._simulate_move(board, available_pieces, row, col, piece)
                        eval, _ = self._minimax(new_board, new_pieces, depth - 1, True)
                        if eval < min_eval:
                            min_eval = eval
                            best_move = (row, col)
            return min_eval, best_move


    def _simulate_move(self, board, available_pieces, row, col, piece):
        """주어진 움직임을 시뮬레이션."""
        new_board = board.copy()
        new_board[row][col] = self.pieces.index(piece) + 1
        new_pieces = available_pieces.copy()
        new_pieces.remove(piece)
        return new_board, new_pieces

    def _is_terminal_state(self, board):
        """현재 상태가 종료 상태인지 확인."""
        # 승리 조건 확인
        for row, col in product(range(4), range(4)):
            if board[row][col] == 0:
                continue
            if self._is_winning_move(row, col):
                return True
        # 모든 칸이 채워지면 종료
        return all(board[row][col] != 0 for row, col in product(range(4), range(4)))

    def _evaluate_board_state(self, board):
        """현재 보드 상태의 점수를 평가."""
        score = 0
        for row, col in product(range(4), range(4)):
            if board[row][col] != 0:
                score += self._evaluate_position(row, col)
        return score
   
