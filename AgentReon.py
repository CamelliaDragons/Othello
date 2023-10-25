from random import choice
from typing import Any, List
from Board import Board,WHITE,BLACK,Color
from Game import Game
from AgentBase import Agent


from tensorflow.keras import  models
import numpy as np


class AgentReon(Agent):
    def __init__(self, color: Color):
        super().__init__(color)
        self.model = models.load_model('AgentReon.h5')
    
    # gridを白の盤面と黒の盤面で分ける
    def split_grid(self,b:Board) -> List[List[Color]]:
        white_board :List[List[Color]] = [[0 for _ in range(8)]for _ in  range(8)]
        black_board :List[List[Color]]= [[0 for _ in range(8)]for _ in  range(8)]
        for i in range(8):
            for j in range(8):
                if b.grid[i][j] == WHITE:
                    white_board[i][j] = 1
                elif b.grid[i][j] == BLACK:
                    black_board[i][j] = 1
        return [white_board, black_board]

    # 可能な行動の中からランダムで行動を選択するエージェント
    def put(self, board) -> List[int]:
        # split_gridで盤面を分けるnpの行列へ
        white_board,black_board = self.split_grid(board)
        if self.color == WHITE:
            split_board = np.reshape(np.array([white_board,black_board],),(1, 2, 8, 8))
        else:
            split_board = np.reshape(np.array([black_board,white_board],),(1, 2, 8, 8))
        split_board = np.reshape(np.array(self.split_grid(board),),(1, 2, 8, 8))

        # 可能な行動の中から, もっとも評価値が高い行動を選択する
        a = self.model.predict(split_board,verbose=0)
        for i in range(64):
            if not [i//8,i%8] in board.get_valid_moves(self.color):
                a[0][i] = 0
        a = np.argmax(a)
        x,y = a//8,a%8
        if [x,y] in board.get_valid_moves(self.color):
            return x, y
        else:
            l = board.get_valid_moves(self.color) # 可能な行動のリストを取得
            print("Warning: AgentReon:put:invalid move")
            x, y = choice(l) # 可能な行動のリストからランダムで選択
            return x,y

from AgentMiniMethod import AgentMiniMethod
REPEAT = 100

p1_win = 0
p2_win = 0
draw = 0
for i in range(REPEAT):
    print("\r", f"Running... {i}/{REPEAT}", end="") # 今何回目かを表示

    g = Game(AgentReon, AgentMiniMethod)
    g.showLog = False # ログを表示しない
    g.showResult = False # 一戦ごとに結果をしない
    winner = g.play()
    if winner == 1:
        p1_win += 1
    elif winner == 2:
        p2_win += 1
    else:
        draw += 1
print(f"player1Win:{p1_win}, player2Win:{p2_win}, Draw:{draw}")


# AgentReon vs AgentRandom 100回
# player1Win:53, player2Win:40, Draw:7
# AgentReon vs AgentMiniMethod 100回
# player1Win:0, player2Win:100, Draw:0
# モデルの性能あ微妙
# 強さはAgentRandomとさほど変わらない
# AgentMiniMethodには全敗
# 考察:
# 1. モデルの学習が不十分
#  正答率や損失関数の値を見たいがmodel.historyがない
#  model.saveしてからload_modelしたものはmodel.historyが消える(Noneになる)ので注意
#  消えたので学習が不十分かどうか検証でいない
#  pickleなどで別途model.historyを保存する必要がある
# 2. predictするときの白と黒の盤面の順番が変？
#  適当に実装したので、白と黒の盤面の順番が変かもしれない
#  後で検証する
# 3. モデルの構造が悪い
#  元のモデルとbias層をちょっと変えた
#  それが悪さしているかもしれない
# 今後:
#  モデルの学習に勝者のデータだけを利用することで性能を向上させられるかもしれない