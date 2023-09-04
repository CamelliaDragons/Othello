from random import choice
from typing import List
from Board import Board, WHITE,BLACK
from AgentBase import Agent
import numpy as np
from itertools import product
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Bias(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.W = tf.Variable(initial_value=tf.zeros((1,64)[1:]), trainable=True)

    def get_config(self):
        base_config = super().get_config()
        return dict(list(base_config.items()))

    def call(self, inputs):
        return inputs + self.W

load_model = keras.models.load_model('./tmp/model-11.h5', custom_objects={'Bias':Bias})

class AgentKoki(Agent):
    def put(self, board) -> List[int]:
        grid = np.array(board.grid_copy(), dtype='int8') #盤面をboardから取得
        #model.predictに渡すためのデータを生成
        newgrid = np.zeros((1,2,8,8))
        for x, y in product(range(8), range(8)):
            if grid[y][x] == self.color:
                newgrid[0][0][y][x] = 1
            elif grid[y][x] == -self.color:
                newgrid[0][1][y][x] = 1
        predicts = load_model.predict(newgrid,verbose=0) #打ち手を取得する
        l = board.get_valid_moves(self.color) # 可能な行動のリストを取得
        #行動可能な打ち手のうち、最善手を取得する
        array = np.zeros((1,64), dtype=np.int8) #行動可能な打ち手
        for i in range(len(l)):
            array[0][l[i][1]*8+l[i][0]] = 1
        for i in range(64):
            predicts[0][i] *= array[0][i]
        xy = np.argmax(predicts)
        x = xy%8
        y = xy//8
        return x, y