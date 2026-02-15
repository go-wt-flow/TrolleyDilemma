#이미지–라벨 매칭 검증 코드
from _04_cnn_training_1 import *

import cv2
import matplotlib.pyplot as plt

# Name list
names = ['_0_forward', '_1_right', '_2_left', '_3_stop']

def display_images(img_path, ax):
    img = cv2.imread(os.path.join(dirname, img_path))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

fig = plt.figure(figsize=(10, 3))
for i in range(4):
    ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
    ax.set_title(names[targets[i+4]], color='blue')
    display_images(files[i+4], ax)
    
plt.show()


