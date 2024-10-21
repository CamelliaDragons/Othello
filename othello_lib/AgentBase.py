from itertools import product
from itertools import chain
from random import choice
from typing import Literal, NewType, List, Type
from abc import ABC, abstractmethod
from othello_lib.Board import Board, WHITE,BLACK

class Agent(ABC):
    def __init__(self, color) -> None:
        """
        初期化メソッド
        Parameters:
            color: 石の色
        """
        self.color = color
    
    @abstractmethod
    def put(self, board:Board) -> List[int]:
        """
        駒を置く場所を返す
        Paramaters:
            board: ボード情報
        Retruns:
            駒を置く場所のリスト[x,y]
        """
        pass