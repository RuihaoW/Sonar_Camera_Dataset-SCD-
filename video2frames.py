# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:59:15 2020

@author: Ruihao Wang
"""
import cv2
import os

s = 'Outdoor_3_a'
dirpath = "D:\\RW\\Sonar_Echo\\Raw_Data\\Outdoor\\Image\\%s\\Raw\\" % s
os.chdir(dirpath)
    
    
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite("frame%d.jpg" % count, image) 
  
        count += 1
        
  
# Driver Code 
if __name__ == '__main__': 
    # Calling the function 
    FrameCapture( dirpath + "GOPR0660.mp4") 
        