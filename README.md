# Othello

このプロジェクトは、Pythonで実装されたオセロゲームです。

## セットアップ
リポジトリをクローンします。

```sh
git clone https://github.com/yourusername/Othello.git
```

プロジェクトはPythonの標準ライブラリのみを使用しています。

## 実行方法
まず、プロジェクトのディレクトリに移動します。
```sh
cd Othello
```

以下のコマンドを実行して、オセロゲームを開始します。
```sh
python Run.py
```

## エージェントの使用方法

プロジェクトには、異なる戦略を持つサンプルエージェントが含まれています。

- `AgentMiniMethod.py`: 相手が置けるマスを最小化するような行動をとるエージェント
- `AgentRandom.py`: ランダムな行動をするエージェント

エージェントを変更するには、`Run.py`内の設定を変更してください。

## ファイル説明
```
Othello/
├── othello_lib/          # オセロのライブラリ
│   ├── AgentBase.py      # エージェントのベースとなるクラス
│   ├── Board.py          # オセロのボードのクラス
│   └── Game.py           # ゲームを進行するクラス
├── AgentMiniMethod.py    # 相手が置けるマスを最小化するような行動をとるエージェント
├── AgentRandom.py        # ランダムな行動をするエージェント
├── Run.py                # 実行方法のサンプルコード
└── .gitignore            # Gitの設定ファイル(無視するファイルを指定)
```

# エージェント作成チュートリアル
新しくファイルを作成して、エージェントを作成する方法を説明します。
まず、適当な名前でファイルを作成します。例えば、`MyAgent.py`とします。
次に、`AgentBase`クラスを継承したクラスを作成します。

```py
from typing import List
from othello_lib.Board import Board, WHITE,BLACK
from othello_lib.AgentBase import Agent

class MyAgent(Agent):
    def put(self, board) -> List[int]:
        pass # ここにエージェントの振る舞いを記述
```

エージェントのふるまいは、`put`メソッドに記述します。
`put`メソッドは、現在のボードの状態を表す`Board`クラスのインスタンスを引数として受け取り、次の一手を返すように実装します。

以下は、ランダムな行動をするエージェントの例です(`RandomAgent.py`から流用)。
```py
class MyAgent(Agent):
    def put(self, board) -> List[int]:
        l = board.get_valid_moves(self.color) # 可能な行動のリストを取得
        x, y = choice(l) 　　　　　　　　　　　 # 可能な行動のリストからランダムで選択
        return x, y
```

`board.get_valid_moves(color)`メソッドは、指定した色のプレイヤーが置けるマスのリストを返します。
その他のメソッドについては、`othello_lib/Board.py`を参照してください。

最後に、`Run.py`内でエージェントを指定して、ゲームを実行します。

```py
from othello_lib.Game import Game
from AgentRandom import AgentRandom
from MyAgent import MyAgent # 作成したエージェントをインポート

game = Game(AgentRandom, AgentMiniMethod)
game.play()
```

## 参考文献
[オセロAIの教科書｜にゃにゃん(山名琢翔)｜note](https://note.com/nyanyan_cubetech/m/m54104c8d2f12)