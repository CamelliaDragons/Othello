from random import choice
from typing import List
from othello_lib.Board import Board, WHITE,BLACK
from othello_lib.AgentBase import Agent



class AgentMiniMethod(Agent):
    # 相手が置ける場所がなくなるように手を打とうとするアルゴリズム
    def put(self, board) -> List[int]:
        l = board.get_valid_moves(self.color) # 可能な行動のリストを取得


        # 自分ができる行動lを全探索して, 相手がおける場所の数を最小化するような行動moveを得る
        move = None
        min = float("inf")
        for i in l:
            # ボードのコピーを得る
            b = board.get_copy()
            
            my_x,my_y = i
            # 自分がiという行動をとったときに, 相手ができる行動の数を取得
            # -self.colorで相手の色を取得できる
            opponent_move_num = len(board.put(my_x,my_y,self.color).get_valid_moves(-self.color))

            if opponent_move_num < min: 
                min = opponent_move_num
                move = i
        
        x, y = move 
        return x, y