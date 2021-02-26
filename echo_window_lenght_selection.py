# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 22:35:42 2021
% This code is used to estimate the echo length in the outdoor echoes. Then
% we can use this length estimate the time-window length when we pick up
% the 64x64 images from echo spectrogram.
@author: wrhsd
"""
import numpy as np
import os
import matplotlib.pyplot as plt


# Go to echo path
s = "filted_echo.npy"
dirpath = "C:\\Users\\wrhsd\\OneDrive\\Desktop\\Run\\Data\\Outdoor\\Echo"
os.chdir(dirpath)
# Load echo (.npy) 
echo = np.load(s)
echo = echo[:-1]
# Echo parameter
(m,n,p) = echo.shape
fs = 400e3
time = np.linspace(0, 1e3*p/fs,p)
dist = (time*340/2)/1e3
# First, choose the last 1000 points from the echo, consider them as noise,
# calculate their std. 
noise_std = []
for i in echo:
    noise = i[:,-1001:-1]
    noise_std = np.append(noise_std, np.std(noise,axis = 1))
# Second, choose the threshold as 10 times of the noise std.
thres = np.mean(noise_std)*10
# Third, find where the ehco crosses the threshold, give a safety margin
# sigma to two sides. ls, rs means left mic starts, right mic strats.
# le, re means left mic ends, right mic ends.
window = np.ndarray((3,m,2), dtype=np.int)
sigma = 1e-3*fs
for i, arg in enumerate(echo):
    ls,rs = np.where(arg[0,2500:]>thres)[0][1], np.where(arg[1,2500:]>thres)[0][1]
    le,re = np.where(arg[0,2500:]>thres)[0][-1], np.where(arg[1,2500:]>thres)[0][-1]
    window[0,i,:] = [ls-sigma+2500,le+sigma+2500]
    window[1,i,:] = [rs-sigma+2500,re+sigma+2500]
    window[2,i,:] = [le-ls,re-rs]
window = window*1e3/fs

# Use boxplot to check how the window length behaves
box = plt.boxplot(window[2,:-1,:])
plt.xticks([1,2],["left","right"])
plt.ylabel("Window length [ms]")
plt.tight_layout()
plt.show()

# Figure out why there are outliers? 
left_outlier = np.where(window[2,:-1,0]>12)   
right_outlier = np.where(window[2,:-1,1]>12) 
    
    
    
    
    