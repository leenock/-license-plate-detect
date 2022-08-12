import numpy as np
import pytesseract
import cv2

#  installed location of Tesseract-OCR in your system -- specify the directory after installing tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

# Load image, create blank mask, convert to HSV, define thresholds, color threshold cproc
image = cv2.imread('YvuWB.png')
result = np.zeros(image.shape, dtype=np.uint8)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0,0,0])
upper = np.array([179,100,130])
mask = cv2.inRange(hsv, lower, upper)
cv2.imshow('mask', mask)

# Perform morphological close and merge for 3-channel ROI extraction
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
extract = cv2.merge([close,close,close])
cv2.imshow('extract', extract)

# Find contours, filter using contour area, and extract using Numpy slicing
cnts = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    area = w * h
    if area < 5000 and area > 2500:
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 3)
        result[y:y+h, x:x+w] = extract[y:y+h, x:x+w]

        # Invert image and throw into Pytesseract
invert = 255 - result
data = pytesseract.image_to_string(invert, lang='eng',config='--psm 6')
print(data)

cv2.imshow('image', image)
cv2.imshow('close', close)
cv2.imshow('result', result)
cv2.imshow('invert', invert)
cv2.waitKey()