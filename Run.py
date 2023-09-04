from Game import Game
from AgentRandom import AgentRandom
from AgentMiniMethod import AgentMiniMethod
from AgentManual import AgentManual
from AgentKoki import AgentKoki


# agentRandomとAgentMiniMethodが一回プレイする

g = Game(AgentRandom, AgentKoki)
g.showLog = True # ログを表示する
g.showResult = True # 一戦ごとに結果を表示する
g.play()


# AgentRandomとAgentMiniMethodが100回プレイする
"""
REPEAT = 100

p1_win = 0
p2_win = 0
draw = 0
for i in range(REPEAT):
    print("\r", f"Running... {i}/{REPEAT}", end="") # 今何回目かを表示

    g = Game(AgentKoki, AgentRandom)
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
"""


# デバッグ用に一手ごとにプレイする
"""
from Board import Board,WHITE,BLACK
b = Board() # 新しいボードを作成
b.show() # 現在のボードを表示
print(b.get_valid_moves(WHITE)) # WHITEが利用可能な行動を表示
b.put(2,3,WHITE) # x=2, y=3に設置
b.show() # 現在のボードを表示
print(b.get_valid_moves(BLACK)) # WHITEが利用可能な行動を表示
b.put(2,2,BLACK) # x=2, y=2に設置
b.show()
# ...
"""
