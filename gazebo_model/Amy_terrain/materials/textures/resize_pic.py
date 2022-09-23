import cv2

img = cv2.imread("terrain2048.png", 0)

img_resized = resized = cv2.resize(img, (515, 515), interpolation = cv2.INTER_AREA)

print(img_resized.shape)

cv2.imwrite("terrain515.png", img_resized)


