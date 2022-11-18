

import cv2

img = cv2.imread("outside.png", 0)

# img_resized = resized = cv2.resize(img, (515, 515), interpolation = cv2.INTER_AREA)

print(img)

cv2.imwrite("outsidef.png", img)


from PIL import Image

import numpy as np
img = Image.open('outsidef.png')

print(img.getbands())