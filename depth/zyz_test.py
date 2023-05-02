image = 'assets/test_image.jpg'
depth = 'assets/test_image_disp.npy'

import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    img = np.load(image)
    dp = np.load(depth)

    print(img)