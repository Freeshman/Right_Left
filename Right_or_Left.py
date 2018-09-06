#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:24:37 2018

@author: hu-tom
"""

import cv2
from pylab import *
def left_right(img1,img2):
    if len(shape(img1))>2:
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)    
    orb=cv2.ORB_create()
    keypoint1, descriptor1 = orb.detectAndCompute(img1, None)
    keypoint2, descriptor2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(descriptor1,descriptor2)
    matches = sorted(matches, key = lambda x:x.distance)
    n=0
    for matche in matches:
        [x1,_]=keypoint1[matche.queryIdx].pt
        [x2,_]=keypoint2[matche.trainIdx].pt
        if x1>x2:
            n=n+1
        else:
            n=n-1
    if n>0:
        return True
    else:
        return False