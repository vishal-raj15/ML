import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import cv2

df = "/home/sensei/Desktop/train"
train_categories = ["humans" , "horses"]

img_size = 80
for cate in train_categories:
    path = os.path.join(df,cate)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        new_data = cv2.resize(img_arr ,(img_size,img_size))
        plt.imshow(new_data , cmap='gray')
        plt.show()
        plt.imshow(img_arr , cmap="gray")
        plt.show()
        break
       

