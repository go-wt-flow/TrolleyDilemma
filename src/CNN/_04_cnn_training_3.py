#CNN 학습용 데이터셋 분할 및 정규화 코드
from _04_cnn_training_1 import *

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

tensors = tensors.astype('float32')/255
targets = to_categorical(targets, 4)

x_train, x_test, y_train, y_test = train_test_split(
        tensors,
        targets,
        test_size = 0.2,
        random_state = 1)

n = int(len(x_test)/2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_valid.shape, y_valid.shape)

