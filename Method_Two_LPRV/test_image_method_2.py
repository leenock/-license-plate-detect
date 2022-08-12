import numpy as np
import pytesseract
import cv2

#  installed location of Tesseract-OCR in your system -- specify the directory after installing tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

# Load image, create blank mask, convert to HSV, define thresholds, color threshold
test_image = cv2.imread('YvuWB.png')
result = np.zeros(test_image.shape, dtype=np.uint8)
hsv = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
lower = np.array([0,0,0])
upper = np.array([179,100,130])
mask = cv2.inRange(hsv, lower, upper)
cv2.imshow('mask', mask)

# Perform morphological close and merge for 3-channel ROI extraction
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
extract = cv2.merge([close,close,close])
cv2.imshow('extract', extract)