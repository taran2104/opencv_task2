import numpy as np
import cv2 as cv
kernel = np.ones((5, 5), np.uint8)
img = cv.imread("lady1.jpg")

imgResize = cv.resize(img, (500, 500))
imgCropped = imgResize[10:400, 100:400]

print(img.shape)
print(imgResize.shape)
print(imgCropped.shape)

imgGray = cv.cvtColor(imgResize, cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(7,7),0)
imgCanny = cv.Canny(imgResize,100,100)
imgDialation = cv.dilate(imgCanny,kernel,iterations=1)
imgEroded = cv.erode(imgDialation,kernel,iterations=1)

cv.imshow("Image",img)
cv.imshow("Gray Image",imgGray)
cv.imshow("Blur Image",imgBlur)
cv.imshow("Canny Image",imgCanny)
cv.imshow("Dilate Image",imgDialation)
cv.imshow("Eroded Image",imgEroded)
cv.imshow("Resized Image", imgResize)
cv.imshow("Cropped Image", imgCropped)

cv.waitKey(0)
