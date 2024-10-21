from random import choice
from typing import List
from othello_lib.Board import Board, WHITE,BLACK
from othello_lib.AgentBase import Agent

class AgentRandom(Agent):
    # 可能な行動の中からランダムで行動を選択するエージェント
    def put(self, board) -> List[int]:
        l = board.get_valid_moves(self.color) # 可能な行動のリストを取得
        x, y = choice(l) # 可能な行動のリストからランダムで選択
        return x, y

