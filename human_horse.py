import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

df = "/home/sensei/Desktop/train"
categories = ["humans" , "horses"]

for cate in categories:
    path = os.path.join(df,cate)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_arr , cmap="gray")
        plt.show()
        break
    break
