#데이터 전처리 
from tensorflow.keras.preprocessing import image as keras_image
import os
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
import pandas as pd

dirname = 'data.1769774079.398169'


def image_to_tensor(img_path):
    # 이미지 불러오기 & 크기 맞추기 ( 120 X 160 크기로 Resize )
    img = keras_image.load_img(
        os.path.join(dirname, img_path),
        target_size=(120,160))
    
    # 숫자 배열 Array 로 변환하기
    x = keras_image.img_to_array(img)
    
    # 차원 늘리기
    return np.expand_dims(x, axis=0)

def data_to_tensor(img_paths):
    list_of_tensors = [
        image_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Load the data
data = pd.read_csv(os.path.join(dirname, "0_road_labels.csv"))
print(data)
data = data.sample(frac=1)

files = data['file']
targets = data['label'].values

tensors = data_to_tensor(files)

print(data.tail())
print(tensors.shape)
print(targets.shape)

