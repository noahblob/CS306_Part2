# reads in all images from dataset, flattens them and puts them into joblib file
import joblib
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pandas as pd
from sklearn.model_selection import train_test_split

datadir = 'images'

flat_data_arr = []
target_arr = []

categories = ['green','red','sheep','speed-up','stop']

for category in categories:
    path = os.path.join(datadir, category)
    print(f'loading... category : {category}')
    for img in os.listdir(path):
        img_arr = imread(os.path.join(path,img))
        img_resized = resize(img_arr, (30,30,3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(categories.index(category))
    print(f'loaded category: {category} successfully')
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target

x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)

joblib.dump((x_train, y_train), 'train_data.joblib')
joblib.dump((x_test, y_test), 'test_data.joblib')