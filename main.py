import os
import cv2
import numpy as np

dir = "C:/omar main/semester 6/DIP/DIP Project/Training_masks/"
count = 0
a = os.listdir(dir)
print(len(a))
my_arr = np.zeros((1200, 256, 256, 3), dtype=np.uint8)
for i in range(len(a)):
    b = os.listdir(os.path.join(dir, a[i]))
    for j in range(len(b)):
        path = os.path.join(dir, a[i], b[j])
        my_img = cv2.imread(path, 1)
        my_arr[count, ...] = my_img
        count += 1

np.save("C:/omar main/semester 6/DIP/DIP Project/train_mask", my_arr)