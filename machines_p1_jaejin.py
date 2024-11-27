import time
import random
from itertools import product
import numpy as np


class P1:  # 동일한 구조로 P2도 작성 가능
    def __init__(self, board, available_pieces):
        self.board = board
        self.available_pieces = available_pieces
        self.timeout = 2  # 제한 시간 (초)

    def _is_winning_move(self, row, col, piece):
        # 해당 위치에 piece를 놓았을 때 승리 조건 확인
        test_board = self.board.copy()
        test_board[row][col] = self._piece_to_index(piece)

        # 승리 조건 (가로, 세로, 대각선, 2x2 그리드) 확인
        return self._check_win(test_board)

    def _blocks_opponent_win(self, row, col):
        # row, col 위치가 상대의 승리를 막는지 확인
        test_board = self.board.copy()
        test_board[row][col] = -1  # 가상의 상대 말 배치
        return self._check_win(test_board)

    def _evaluate_position(self, row, col):
        # 단순한 평가 함수: 중앙 위치를 선호하거나 경계선에 가까운 위치의 점수를 낮게 부여
        score = 0
        if row == col or row + col == len(self.board) - 1:  # 대각선
            score += 2
        if (row, col) == (1, 1):  # 중앙 위치
            score += 3
        return score

    def _get_empty_squares(self):
        # 현재 보드에서 빈 칸 (row, col)을 모두 반환
        empty_squares = []
        for row in range(len(self.board)):
            for col in range(len(self.board[0])):
                if self.board[row][col] == 0:
                    empty_squares.append((row, col))
        return empty_squares

    def _piece_to_index(self, piece):
        # piece를 해당하는 인덱스 값으로 변환
        return self.available_pieces.index(piece) + 1

    def _check_win(self, board):
    # 보드의 승리 조건 확인
        def check_line(line):
            if 0 in line:
                return False
            characteristics = []
            for piece_idx in line:
                # piece_idx가 1 이상이고 self.available_pieces의 범위를 벗어나지 않도록 체크
                if piece_idx > 0 and piece_idx <= len(self.available_pieces):
                    characteristics.append(self.available_pieces[piece_idx - 1])
                else:
                    return False  # 잘못된 piece_idx가 발견되면 바로 False 반환
            for i in range(4):  # 특성 비교
                if len(set(characteristics[i])) == 1:
                    return True
            return False

        # 가로, 세로, 대각선 확인
        for col in range(4):
            if check_line([board[row][col] for row in range(4)]):
                return True
        for row in range(4):
            if check_line([board[row][col] for col in range(4)]):
                return True
        if check_line([board[i][i] for i in range(4)]) or check_line([board[i][3 - i] for i in range(4)]):
            return True

        # 2x2 그리드 확인
        for r in range(3):
            for c in range(3):
                subgrid = [board[r][c], board[r][c + 1], board[r + 1][c], board[r + 1][c + 1]]
                if 0 not in subgrid:
                    characteristics = []
                    for idx in subgrid:
                        if idx > 0 and idx <= len(self.available_pieces):
                            characteristics.append(self.available_pieces[idx - 1])
                        else:
                            return False  # 잘못된 piece_idx가 발견되면 바로 False 반환
                    for i in range(4):  # 특성 비교
                        if len(set(characteristics[i])) == 1:
                            return True
        return False


    def _minimax(self, depth, is_maximizing, alpha, beta, selected_piece):
        if depth == 0 or self._check_win(self.board):
            return self._evaluate_board(selected_piece)

        empty_squares = self._get_empty_squares()
        if is_maximizing:
            max_eval = float('-inf')
            for row, col in empty_squares:
                self.board[row][col] = self._piece_to_index(selected_piece)
                eval = self._minimax(depth - 1, False, alpha, beta, selected_piece)
                self.board[row][col] = 0
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # 가지치기
            return max_eval
        else:
            min_eval = float('inf')
            for row, col in empty_squares:
                self.board[row][col] = -1  # 상대방의 가상 말
                eval = self._minimax(depth - 1, True, alpha, beta, selected_piece)
                self.board[row][col] = 0
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # 가지치기
            return min_eval


    def _evaluate_board(self, selected_piece):
        # 보드의 전체 점수를 평가
        score = 0

        # 1. 각 칸의 말을 평가
        for row in range(4):
            for col in range(4):
                if self.board[row][col] == 0:  # 빈 칸은 건너뜀
                    continue
                
                piece_index = self.board[row][col] - 1  # 보드 값은 1부터 시작하므로 -1 필요
                if piece_index < 0 or piece_index >= len(self.available_pieces):  # 유효성 체크
                    continue  # 잘못된 인덱스가 있으면 무시

                piece = self.available_pieces[piece_index]

                # 대각선/중앙 선호
                if row == col or row + col == 3:  # 대각선 위치
                    score += 2
                if (row, col) == (1, 1):  # 중앙 위치
                    score += 3

        # 2. 상대방의 승리 방지 점수 추가
        empty_squares = self._get_empty_squares()
        for row, col in empty_squares:
            if self._blocks_opponent_win(row, col):
                score -= 10  # 상대방 승리를 막는 중요성 반영

        return score

    def _is_winning_piece(self, piece):
    # 특정 말을 사용했을 때 승리 가능성 확인
        empty_squares = self._get_empty_squares()
        for row, col in empty_squares:
            test_board = self.board.copy()
            test_board[row][col] = self._piece_to_index(piece)
            if self._check_win(test_board):
                return True
        return False

    def _evaluate_piece(self, piece):
        score = 0

        # 1. 희소한 특성을 가진 말 선호
        for i in range(4):  # 각 특성에 대해
            characteristic_count = sum(p[i] for p in self.available_pieces)
            if piece[i] == 1:
                score += len(self.available_pieces) - characteristic_count
            else:
                score += characteristic_count

        # 2. 중앙 및 대각선에 유리한 말을 선호
        empty_squares = self._get_empty_squares()
        for row, col in empty_squares:
            test_board = self.board.copy()
            test_board[row][col] = self._piece_to_index(piece)
            if self._check_win(test_board):
                score += 10  # 승리에 기여할 가능성이 높은 말 선호

        # 3. 상대방의 선택 제한
        opponent_impact = 0
        for opp_piece in self.available_pieces:
            if opp_piece == piece:
                continue
            if self._is_winning_piece(opp_piece):  # 상대방 승리에 기여할 수 있는 말인지
                opponent_impact -= 5  # 상대방에게 불리한 말을 선택
        score += opponent_impact

        return score

    def select_piece(self):
        start_time = time.time()
        best_piece = None
        best_score = float('-inf')

        # 가능한 말을 모두 평가
        for piece in self.available_pieces:
            if time.time() - start_time > self.timeout:
                break  # 시간 초과 시 중단

            # 1. 상대방이 승리할 수 있는 말을 피함
            if self._is_winning_piece(piece):
                continue

            # 2. 말의 전략적 가치를 평가
            score = self._evaluate_piece(piece)
            if score > best_score:
                best_score = score
                best_piece = piece

        # 3. 제한 시간 초과 또는 모든 말 평가 후 최선의 선택 반환
        if best_piece:
            return best_piece

        # 4. 아무 것도 선택하지 못했다면 임의의 말 반환
        return random.choice(self.available_pieces)
    
    def place_piece(self, selected_piece):
        start_time = time.time()
        empty_squares = self._get_empty_squares()
        best_move = None
        best_value = float('-inf')

        # 1. 우선, 즉시 승리할 수 있는 위치 찾기
        for row, col in empty_squares:
            if self._is_winning_move(row, col, selected_piece):
                return row, col

        # 2. 상대방의 승리를 즉시 방어할 수 있는 위치 찾기
        for row, col in empty_squares:
            if self._blocks_opponent_win(row, col):
                return row, col

        # 3. 제한 시간 동안 Iterative Deepening으로 최적 수 찾기
        depth = 1
        while time.time() - start_time < self.timeout:
            for row, col in empty_squares:
                self.board[row][col] = self._piece_to_index(selected_piece)
                value = self._minimax(depth, False, float('-inf'), float('inf'), selected_piece)
                self.board[row][col] = 0

                if value > best_value:
                    best_value = value
                    best_move = (row, col)
            depth += 1

        # 4. 최적 수 반환
        if best_move:
            return best_move

        # 5. 시간 초과 시 임의의 빈 칸 반환
        return random.choice(empty_squares)


