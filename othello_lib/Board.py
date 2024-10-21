from itertools import product
from itertools import chain
from random import choice
from typing import Literal, NewType, List, Type
from abc import ABC, abstractmethod
from copy import deepcopy
Color = NewType("Color", int)

EMPTY = Color(0)
WHITE = Color(1)
BLACK = Color(-1)


class Board():
    def __init__(self):
        self.grid : List[List[Color]] = [[EMPTY for _ in range(8)]for _ in  range(8)]
        self.grid[3][4] = WHITE
        self.grid[4][3] = WHITE
        self.grid[3][3] = BLACK
        self.grid[4][4] = BLACK
    
    def show(self) -> None:
        """
        盤面を表示する.
        """
        print("  01234567")
        for index, row in enumerate(self.grid):
            print(index, end=" ")
            for cell in row:
                print(self._get_cell_symbol(cell), end="")
            print()

    def _get_cell_symbol(self, cell: Color) -> str:
        symbols = {EMPTY: ".", WHITE: "o", BLACK: "x"}
        return symbols.get(cell, "+")
    
    def put(self, x: int, y: int, c: Color) -> 'Board':
        """
        x,yにpの色の石を置く.

        Parameters:
            x: x座標
            y: y座標
            c: 石の色
        """
        flipped_pieces = self.get_flipped_pieces(x, y, c)
        self.grid[y][x] = c
        for x_flipped, y_flipped in flipped_pieces:
            self.grid[y_flipped][x_flipped] = c
        return self
            
    def get_flipped_pieces(self, x: int, y: int, c: Color) -> List[List[int]]:
        """
        x,yにpの色の石を置いた時にひっくり返る石の座標を返す.
        
        Parameters:
            x: x座標
            y: y座標
            c: 石の色
        Returns:
            ひっくり返る石の座標のリスト
        """
        flipped_pieces = []
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            tx, ty = x, y
            pieces_to_flip  = []
            if dx == 0 and dy == 0:
                continue
            while True:
                tx += dx
                ty += dy
                if tx < 0 or ty < 0 or tx >= 8 or ty >= 8:
                    break
                if self.grid[ty][tx] == EMPTY:
                    break
                elif self.grid[ty][tx] == c:
                    flipped_pieces.extend(pieces_to_flip)
                    break
                elif self.grid[ty][tx] == -c:
                    pieces_to_flip.append([tx, ty])
        
        return flipped_pieces
    
    def get_valid_moves(self, c: Color) -> List[List[int]]:
        """
        cの色の石を置ける座標のリストを返す.

        Parameters:
            c: 石の色
        Returns:
            石を置ける座標のリスト
        """
        valid_moves = []
        
        for x, y in product(range(8), range(8)):
            if self.grid[y][x] != EMPTY:
                continue
            
            flipped_pieces = self.get_flipped_pieces(x, y, c)
            if len(flipped_pieces) > 0:
                valid_moves.append([x, y])
        
        return valid_moves

        
    
    def isGameEnd(self) -> bool:
        """
        ゲームが終了しているかどうかを返す.
        Returns:
            ゲームが終了しているかどうか
        """
        if (not self.has_valid_move(WHITE)) and  (not self.has_valid_move(BLACK)):
            return True
        else:
            return False
    
    def winner(self) -> int:
        """
        ゲームが終了しているかどうかを返す.
        Returns:
            プレイヤー1が勝利したなら1 
            プレイヤー2が勝利したなら2
            引き分けなら0
        """
        player1_num = list(chain(*self.grid)).count(1)
        player2_num = list(chain(*self.grid)).count(-1)
        if player1_num > player2_num:
            return 1
        elif player1_num < player2_num:
            return 2
        else:
            return 0
    def count(self) -> List[int]:
        """
        プレイヤーの駒の数をそれぞれ返す
        """
        player1_num = list(chain(*self.grid)).count(1)
        player2_num = list(chain(*self.grid)).count(-1)
        return player1_num, player2_num
        
    def has_valid_move(self,p) -> bool:
        """
        pの色の石を置ける場所があるかどうかを返す.
        Parameters:
            p: 石の色
        Returns:
            石を置ける場所があるかどうか 
        """
        return len(self.get_valid_moves(p)) != 0
    
    def get_copy(self) -> 'Board':
        """
        自身のコピーを返す
        """
        return deepcopy(self)