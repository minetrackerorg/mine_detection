# 
# Created in Óbuda University, BioTech lab, 2023
#
# Written by Bence Biricz, Sándor Burian, Miklós Vincze, Abdallah Benhamida, based on:
#   - https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
#   - https://stackoverflow.com/a/45338831/7973735
#   - https://scikit-image.org/docs/stable/auto_examples/filters/plot_entropy.html
#
# Marine mine detection algorithm
# the output is in "#timestamp class instance centroid_x centroid_y" format
# also with removing comments you will be able to see the results.
#
import sys
import threading
import cv2
import matplotlib.pyplot as plt
import time
import skimage.color
import skimage.io
import random
import time
import PIL
from PIL import Image
import scipy.ndimage as sn
import imageio.v2 as imageio
import IPython.display
from skimage.util import img_as_float
##########
from skimage import data
from skimage import data_dir
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import io
import os
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import data, color
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

from cv2 import ROTATE_90_CLOCKWISE
import numpy as np
import pytesseract
from pytesseract import Output
import calendar
import time

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

current_GMT = time.gmtime()

from matplotlib.widgets import Slider
one = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_1 = Slider(one, 'Val1', 0.0, 255.0, 150)
two = plt.axes([0.25, 0.1, 0.65, 0.03])
slider_2 = Slider(two, 'Val2', 0.0, 255.0, 200)

rotation_counter = 0
max_rotation = 4

desired_1 = "1\n"
desired_2 = "2\n"
desired_3 = "3\n"
desired_4 = "4\n"
desired_5 = "5\n"
desired_6 = "6\n"

# Detecting box
opi_found = False
pixel_count_cropped_im = 0
red_backround_number = False
back_over_yellow_number = False
class_found = ""
object_found = ""
centroid_x = ""
centroid_y = ""
red_marker_found = False
detected_number = ""

def number_find_red_background(img,one, two):
    rotation_counter = 0
    while True:
        # Cvt to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #cv2.imshow("hsv", hsv)

        # Get binary-mask
        #fekete fehérnél: 57, 153
        msk = cv2.inRange(hsv, np.array([0, int(one), 0]), np.array([255, int(two), 255]))
        #msk = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([255, 255, 178]))
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dlt = cv2.dilate(msk, krn, iterations=1)
        thr = 255 - cv2.bitwise_and(dlt, msk)

        # OCR
        detected_number = pytesseract.image_to_string(
            thr, config="--psm 10 --oem 3 -c tessedit_char_whitelist=3456")
        #img = cv2.rotate(img, ROTATE_90_CLOCKWISE)
        # cv2.waitKey(0)
        rotation_counter += 1

        #cv2.imshow("msk", msk)
        cv2.imshow("thr", thr)

        # red background image found --> class 3
        if detected_number == desired_3:
            detected_number = "3"
            print("Number found: ")
            print(detected_number)
            #return detected_number
            break
        elif detected_number == desired_4:
            detected_number = "4"
            print("Number found: ")
            print(detected_number)
            #return detected_number
            break
        elif detected_number == desired_5:
            detected_number = "5"
            print("Number found: ")
            print(detected_number)
            #return detected_number
            break
        elif detected_number == desired_6:
            detected_number = "6"
            #return detected_number
            print("Number found: ")
            print(detected_number)
            break
        elif rotation_counter == max_rotation:
            #detected_number = "?"
            #return detected_number
            break

def black_number(img):
    rotation_counter = 0
    # detect black number over yellow pipes
    while True:
        # Cvt to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Get binary-mask
        msk = cv2.inRange(hsv, np.array(
            [0, 0, 155]), np.array([255, 255, 178]))
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dlt = cv2.dilate(msk, krn, iterations=1)
        thr = 255 - cv2.bitwise_and(dlt, msk)

        # OCR
        detected_number = pytesseract.image_to_string(
            thr, config="--psm 10 --oem 3 -c tessedit_char_whitelist=123456789")
        img = cv2.rotate(img, ROTATE_90_CLOCKWISE)
        # cv2.waitKey(0)
        rotation_counter += 1

        # yellow background black image found --> class 2
        if detected_number == desired_1:
            detected_number = "1"
            print("Black number: ",detected_number)
            #return detected_number
            break
        elif detected_number == desired_2:
            detected_number = "2"
            print("Black number: ",detected_number)
            #return detected_number
            break
        elif detected_number == desired_3:
            detected_number = "3"
            print("Black number: ",detected_number)
            #return detected_number
            break
        elif detected_number == desired_4:
            detected_number = "4"
            print("Black number: ",detected_number)
            #return detected_number
            break
        elif rotation_counter == max_rotation:
            #rotation_counter = 0
            #detected_number = ""
            #return detected_number
            break


def rgb2gray(rgb):
        return np.dot(rgb[:,:,0:3], [0.2989, 0.5870, 0.1140])

def rgb_as_gray(rgb):
    return rgb[:,:,0]

def entropy_of_image(image):
    image_gray = rgb2gray(image)
    f_size = 20
    radi = list(range(1,10))
    #fig, ax = plt.subplots(3,3,figsize=(15,15))
    #for n, ax in enumerate(ax.flatten()):
    #    ax.set_title(f'Radius at {radi[n]}', fontsize = f_size)
    entropy_tmp = entropy(image_gray, disk(radi[8])) 
    #   #ax.imshow(entropy_tmp, cmap = 'magma');
    #   ax.imshow(entropy_tmp, cmap = 'magma');
    #   ax.set_axis_off()
    #fig.tight_layout()
    plt.savefig('images/tmpentropy.jpg')  
    return entropy_tmp  

def entropycalculator(image):    
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 4),
                                sharex=True, sharey=True)

    img0 = ax0.imshow(image, cmap=plt.cm.gray)
    ax0.set_title("Image")
    ax0.axis("off")
    fig.colorbar(img0, ax=ax0)

    img1 = ax1.imshow(entropy(image, disk(5)), cmap='gray')
    ax1.set_title("Entropy")
    ax1.axis("off")
    fig.colorbar(img1, ax=ax1)

    fig.tight_layout()

    plt.show()    

def buoydetect(default_file):
    #default_file = 'images/pic51.jpg'
    
    #filename = os.path.join(default_file)
    #img = io.imread(filename)
    img = default_file
    
    #img = img_as_ubyte(img)
    img = img_as_float(img)
    ##gray_entropy = entropycalculator(img)
    #gray_entropy = entropy_of_image(img)


    # #Image.open('images/tmpentropy.png').save('images/tmpentropy.jpg','JPEG')
    #default_file = 'images/tmpentropy.jpg'

    #filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    #src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
    #src = cv2.imread(default_file, cv2.IMREAD_COLOR)
    src = img
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
   
    #print(src)
    #print("*******")
    
    #print(src)
    #img1 = skimage.color.rgb2gray(skimage.io.imread(filename)[:,:,:3]) * 255
    img1 = skimage.color.rgb2gray(default_file[:,:,:3]) * 255
    #img2 = np.array(Image.open(filename).convert('L'))
    img2 = img
    #img3 = imageio.imread(filename)
    img3 = default_file
    gray = lambda rgb : np.dot(img3[... , :3] , [0.21 , 0.72, 0.07]) 
    img3 = gray(img3)
    img_diff = np.ndarray(shape=img1.shape, dtype='float32')
    img_diff.fill(128)
    img_diff += (img1 - img3)
    img_diff -= img_diff.min()
    img_diff *= (255/img_diff.max())
    plt.imshow(np.array(PIL.Image.fromarray(img_diff).convert('RGB')))
    #plt.savefig('images/tmpdif.png')

    gray=rgb_as_gray(np.array(PIL.Image.fromarray(img_diff).convert('RGB')))

    #cv.imshow('', gray)
    #cv.waitKey(0)
    
    #print(gray)
    #print("*******")
    gray = cv2.medianBlur(gray, 5)
    #cv.imshow('', gray)
    #cv.waitKey(0)
    
    #cv2.imshow("gray", gray)

    rows = gray.shape[0]
                            
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100,
                        param1=100, param2=44,
                        minRadius=0, maxRadius=600)

    #cv2.circle(image, center_coordinates, radius, color, thickness)
    x,y,r = 0, 0, 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            if(r < i[2]):
                x = i[0]
                y = i[1]
                r = i[2]
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (0, 255, 0), 3)

    #cv2.imshow("Video", src)
    
    center = (x, y)
    # circle center
    #cv.circle(src, center, 1, (0, 100, 100), 3)
    # circle outline
    radius = r
   


    im_height, im_width, _ = src.shape

    im_center_x = im_width/2
    im_center_y = im_height/2

    output = ""
    #if(circles is None):
       # print("straight")
    if(r>100 and r<150):
        #print("stop")
        if(x<im_center_x):
            print("right")
            output = "right"
        else:
            print("left")
            output = "left"
    elif(circles is not None and r<100):
        print("straight")
        output = "forward"
    elif(r>150):
        print("stop")
        output = "stop"


    ts = time.time()
    if(circles is not None):
        tmpstr = str(ts) + ", " + ", X:" + str(x)+ ", Y:" + str(y) + " , R:" + str(r)
        #print(ts, "1", color, x, y)
        print(tmpstr)
        tmpstr = str(tmpstr)
    #return src
    output = str(output)
    #cv2.imshow("detected circles", src)
    #return output
    return src

# Face recognition and opencv setup
cap = cv2.VideoCapture("test_2.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, size)
#cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")


def update1(val1):
                val1 = slider_1.val
                return str(val1)
def update2(val2):
                val2 = slider_2.val
                return str(val2)

slider_1.on_changed(update1)
slider_2.on_changed(update2)

val1 = 57
val2 = 153

def task1():
    # Read until video is completed
    while(cap.isOpened()):       
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:


        # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                #t1.join()
                break
    # Break the loop
        else:
            break
    # When everything done, release
    # the video capture object
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()

 # Read until video is completed
while(cap.isOpened()):

# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        img = buoydetect(frame)
        
        cv2.imshow('frame', img)        
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break


# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
