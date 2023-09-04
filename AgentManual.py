from random import choice
from typing import List
from Board import Board, WHITE,BLACK
from AgentBase import Agent

class AgentManual(Agent):
    # 可能な行動の中からどれかを手動で選択するエージェント
    def put(self, board) -> List[int]:
        l = board.get_valid_moves(self.color) # 可能な行動のリストを取得
        x, y = -1, -1
        while [x, y] not in l:
            print(f"candidates:{l}")
            print("x:", end=" ")
            x = int(input())
            print("y:", end=" ")
            y = int(input())
        return x, y