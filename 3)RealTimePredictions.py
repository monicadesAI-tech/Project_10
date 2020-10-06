#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Required libraries
import cv2
import pickle
import numpy as np
import pandas as pd


# In[8]:


# parameters for OpenCV putText function
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (250,250)
fontScale              = 1
fontColor              = (255,0,0)
lineType               = 2

# Loading trained XGBoost classifier
clf = pickle.load(open('model/XGBmodel.sav', 'rb'))
    
# Function to classify solid waste
def classify(img):
    # Converting image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (512, 384))
    # Calculating histograms
    hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
    hist = hist.reshape(1,hist.shape[0])
    # Predicting waste category
    k = clf.predict(np.array(hist))
    # Writing predicted class on frame
    img = cv2.putText(img, k[0], bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    return img

# Creating object to capture video using webcam
video_capture = cv2.VideoCapture(0)
while True:
    # Capturing video
    _, frame = video_capture.read()
    frame_copy = frame.copy()
    # Passing individual frames of video
    canvas = classify(frame)
    # Showing frame returned from classify function
    cv2.imshow('Video', canvas)
    # Press q to exit webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('5.jpg', frame_copy)
        break
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




