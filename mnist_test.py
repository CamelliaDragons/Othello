import time
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

start_time = time.time()  # スタート時間計測

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


history = model.fit(x_train, y_train,
                    # batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[EarlyStopping(patience=1, verbose=1)])

score = model.evaluate(x_test, y_test, verbose=0)
print('loss:', score[0])
print('accuracy:', score[1])

end_time = time.time() - start_time  # 終了時間を計測
print("学習時間：", str(round(end_time, 3)), "秒でした。")
