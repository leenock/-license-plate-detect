import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image, ImageTk
from imutils.contours import sort_contours

#  installed location of Tesseract-OCR in your system -- specify the directory after installing tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'


# change root background col and ect
def center_window(width=300, height=200):
    # get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))


root = tk.Tk()
root.attributes('-alpha', 2)  # transparency
root.iconbitmap('./pythontutorial.ico')
root.configure(bg="white")
root.title("Car Plate Number Detection System")

# create a menubar
menubar = Menu(root)
root.config(menu=menubar)

# create a menu
file_menu = Menu(menubar)


# function to choose image from the local directory
def myClick():
    link = askopenfilename()
    my_img = ImageTk.PhotoImage(Image.open(link))
    my_label.configure(image=my_img)
    my_label.image = my_img


my_label = Label()
my_label.pack()

# add a menu item to the menu
file_menu.add_command(
    label='Click To choose a Vehicle', command=myClick
)

file_menu.add_command(
    label='Click To Exit Program',
    command=root.destroy

)

# add the File menu to the menubar
menubar.add_cascade(
    label="Click to View Program Menu",
    menu=file_menu
)


# Function to scan the vehicle image for plate number detection.

def myimage():
    link = askopenfilename()
    my_img = ImageTk.PhotoImage(Image.open(link))
    my_label.configure(image=my_img)
    my_label.image = my_img

    # ***************************************************************************************#

    # reading the image that has been selected
    img1 = cv2.imread(link, 1)

    # ***************************************************************************************#

    # image is resized keeping aspect ratio same

    height = img1.shape[0]
    width = img1.shape[1]

    scale_factor = 0.8
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    dimensions = (new_width, new_height)

    new_image = cv2.resize(img1, dimensions, interpolation=cv2.INTER_LINEAR)
    # print("New shape:   ", new_image.shape)

    cv2.imshow("Original vehicle image", img1)
    cv2.imshow("Resized vehicle image", new_image)

    # ***************************************************************************************#

    # changing the image to gray scale

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv2.imshow('1 - Grayscale vehicle Image Conversion', gray)

    # ***************************************************************************************#

    # first method of noice reduction
    # Image Smoothing techniques help in reducing the noise.
    # bilateral filter is used for smoothening images and reducing noise
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    cv2.imshow('1 - bilateral filter is used for smoothening images and reducing noise', bfilter)

    # Another way--> Second method of noice reduction
    # Image Smoothing techniques help in reducing the noise using GaussianBlur To reduce the Noise
    # filtered = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow('GaussianBlur To reduce the Noise', filtered)

    # ***************************************************************************************#

    # Edge detection method is used to detect edges
    edged = cv2.Canny(bfilter, 10, 100)
    cv2.imshow('Applying edges for localization', edged)

    # ***************************************************************************************#
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hir = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    plate = False
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            if 2.5 < w / h < 4.1:
                plate = True
                cv2.drawContours(img1, c, -1, (0, 255, 0), 3)  # draw contours on the original image img1
                cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 3)
                break

    if not plate:
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(c)
                if 2.5 < w / h < 4.5 and 10000 <= (w * h):
                    plate = True
                    cv2.drawContours(img1, c, -1, (0, 0, 255), 1)
                    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    break
                    # show the original image
    cv2.imshow('Original Image', img1)

    if plate:
        cropped = img1[y - 10:y + h + 25, x - 10:x + w + 30]
        # show the cropped image after finding contours from the original image
        cv2.imshow('Original Image plate number based on Contours', cropped)

    # ***************************************************************************************#

    #  Applying mask
    mask = np.zeros(gray.shape, np.uint8)
    masked_plate_number = cv2.drawContours(mask, [approx], 0, 255, -1)
    masked_plate_number = cv2.bitwise_and(img1, img1, mask=mask)
    cv2.imshow('Applying mask on plate number', masked_plate_number)

    # ***************************************************************************************#

    # Enhancing the number plate with histogram equalisation before further processing
    y, cr, cb = cv2.split(cv2.cvtColor(masked_plate_number, cv2.COLOR_RGB2YCrCb))
    # dividing the three channels and converting the image to a YCrCb model
    y = cv2.equalizeHist(y)
    # Applying histogram equalisation
    final_image = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
    # unifying the three channels
    cv2.imshow("Enhanced Number Plate", final_image)

    # ***************************************************************************************#

    # cropping the image after mask preprocessing
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    final_image = gray[x1:x2 + 1, y1:y2 + 1]
    cv2.imshow('cropped number plate after masking', final_image)

    # ***************************************************************************************#

    # Threshold using Otsu's --  threshold the distance transform using Otsu's method

    work_img = cv2.threshold(final_image, 0, 255, cv2.THRESH_OTSU)[1]
    cv2.imshow('Threshold', work_img)
    # cv2.imwrite('Threshold.png', work_img)

    # ***************************************************************************************#
    # segmentation of the image placing each letter in a rectangular box code:

    ret, thresh2 = cv2.threshold(final_image, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(thresh2, cv2.MORPH_DILATE, kernel)
    bboxes = []
    bboxes_img = final_image.copy()
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(bboxes_img, (x, y), (x + w, y + h), (0, 0, 0), 1)
        bboxes.append((x, y, w, h))

    # cv2.imshow('Image', bboxes_img)

    # ***************************************************************************************#

    # pytesseract - converting the image cropped to text: -
    #  text = pytesseract.image_to_string(cropped_image, config='--psm 11')
    # print("Detected license plate Number is:", text)
    invert = 255 - work_img
    custom_config = r'--oem 3 -c itemised_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz --psm 6'
    data = pytesseract.image_to_string(invert, lang='eng', config=custom_config)
    print("Detected license plate Number is:", data)

    # ***************************************************************************************#


my_label = Label()
my_label.pack()

# add the File menu to the menubar
menubar.add_cascade(
    label="Click to Scan Vehicle Plate Number", command=myimage

)

center_window(1000, 640)
root.mainloop()
cv2.destroyAllWindows()

# ***************************************************************************************#
# END OF THE PROGRAM