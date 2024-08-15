import os
import cv2
import numpy as np

labels_dictionary = {
    0: [115, 0, 108],
    1: [122, 1, 145],
    2: [148, 47, 216],
    3: [242, 246, 254],
    4: [130, 9, 181],
    5: [157, 85, 236],
    6: [106, 0, 73],
    7: [168, 123, 248],
    8: [0, 0, 0],
    9: [255, 255, 127],
    10: [142, 255, 127],
    11: [127, 127, 255]}


def label_the_image(image):
    r, c, _ = image.shape
    labeled_image = np.full((r, c), 8, dtype=np.uint8)  # Initialize with 9
    label_keys = labels_dictionary.keys()
    for i, key in enumerate(label_keys):
        color = labels_dictionary[key]
        mask = np.all(image == color, axis=2)
        labeled_image[mask] = i
    return labeled_image


def decode_image(labeled_image):
    r, c = labeled_image.shape[:2]
    decoded_image = np.zeros((r, c, 3), dtype=np.uint8)
    label_keys = labels_dictionary.keys()
    for i in label_keys:
        color = labels_dictionary[i]
        decoded_image[labeled_image == i] = color
    return decoded_image


img = cv2.imread('Training_masks/BCC/BCC_1_0001.png')
cv2.imshow("original image", img)
cv2.waitKey(0)
img1 = label_the_image(img)
img2 = decode_image(img1)
image_array = np.array(img)

cv2.imshow("decoded image", img2)
cv2.waitKey(0)
