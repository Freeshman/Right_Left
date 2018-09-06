from pylab import *
import cv2
from Right_or_Left import left_right
img1 = cv2.imread('left.png',0)
img2 = cv2.imread('right.png',0)
order121=left_right(img1,img2)
order122=left_right(img2,img1)
print(order121)
print(order122)
if order121:
    print('The img1 is captured from the left side')
    imgl=img1
    imgr=img2
else:
    print('The img2 is captured from the left side')
    imgl=img2
    imgr=img1
stereo = cv2.StereoBM_create( 16, 15)
disparity = stereo.compute(imgl, imgr)
imshow(disparity,'gray')
