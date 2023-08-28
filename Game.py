from itertools import product
from itertools import chain
from random import choice
from typing import Literal, NewType, List, Type
from abc import ABC, abstractmethod
from AgentBase import Agent
from Board import Board, WHITE,BLACK


class Game:
    def __init__(self,player1,player2) -> None:
        self.players: List[Agent] = [player1(WHITE), player2(BLACK)]
        self.board: Board = Board()
        self.showLog = False # 一手ごとに盤面を表示
        self.showResult = False # 終了時に結果を表示

    def play(self) -> int:
        # ゲームをプレイして勝者を返す
        while not self.board.isGameEnd():
            for player in self.players:
                if self.showLog:
                    self.board.show()
                color = player.color
                if self.board.has_valid_move(color):
                    x, y = player.put(self.board.get_copy())
                    self.board.put(x, y, color)
        if self.showResult:
            self.board.show()
            p1_count,p2_count = self.board.count()
            print(f"player1:{p1_count}, player2:{p2_count}")
            winner = self.board.winner()
            print(f"winner:{winner}")

        return self.board.winner()