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
s = "filted_echo_Outdoor_9_a.npy"
dirpath = "C:\\Users\\wrhsd\\OneDrive\\Desktop\\Run\\Data\\Outdoor\\Echo"
os.chdir(dirpath)
# Load echo (.npy) 
echo = np.load(s)
echo = echo[:-1]
echo[:,:,0:40] = 0
# Echo parameter
(m,n,p) = echo.shape
fs = 400e3
time = np.linspace(0, 1e3*p/fs,p)
dist = (time*340/2)/1e3
# First, choose the last 1000 points from the echo, consider them as noise,
# calculate their std. 
noise_std = []
for i in echo:
    noise = i[:,2500:3500]
    noise_std = np.append(noise_std, np.std(noise,axis = 1))
# Second, choose the threshold as 10 times of the noise std.
thres = np.mean(noise_std)*10
# Third, find where the ehco crosses the threshold, give a safety margin
# sigma to two sides. ls, rs means left mic starts, right mic strats.
# le, re means left mic ends, right mic ends.
window = np.ndarray((3,m,2), dtype=np.int)
sigma = 1e-3*fs
for i, arg in enumerate(echo):
    left_check = np.absolute(arg[0,2500:])>thres
    right_check = np.absolute(arg[1,2500:])>thres
    if left_check.any():
        ls,le = np.where(np.absolute(arg[0,2500:])>thres)[0][0],\
            np.where(np.absolute(arg[0,2500:])>thres)[0][-1]
    else: 
        ls,le = 800,0
    if right_check.any():
        rs,re = np.where(np.absolute(arg[1,2500:])>thres)[0][0],\
           np.where(np.absolute(arg[1,2500:])>thres)[0][-1] 
    else:
        rs,re = 800, 0
        
    window[0,i,:] = [ls-sigma+2500,le+sigma+2500]
    window[1,i,:] = [rs-sigma+2500,re+sigma+2500]
    window[2,i,:] = [le-ls+2*sigma,re-rs+2*sigma]
window = window*1e3/fs

# Use boxplot to check how the window length behaves
box = plt.boxplot(window[2,:-1,:])
plt.xticks([1,2],["left","right"])
plt.ylabel("Window length [ms]")
plt.title('_'.join(s.split('_')[1:4]))
plt.tight_layout()
plt.show()

# Figure out why there are outliers? 
l = plt.boxplot(window[2,:-1,0])
r = plt.boxplot(window[2,:-1,1])
left_outlier = l["fliers"][0].get_data()[1]
right_outlier = r["fliers"][0].get_data()[1]

left_idx = [np.where(window[2,:,0] == i) for i in left_outlier]
right_idx = [np.where(window[2,:,1] == i) for i in right_outlier]

for i in left_idx:
    for j in i:
        print('left mic')
        print(j)
        plt.figure()
        ax1 = plt.subplot(211)
        ax1.plot(time,echo[j[0],0,:])
        ax1.hlines(y=thres,xmin=10,xmax=25,color='r',linestyle='--')
        ax1.hlines(y=-thres,xmin=10,xmax=25,color='r',linestyle='--')
        ax1.set_ylim((-0.5,0.5))
        ax1.set_ylabel("Amplitude")
        ax1.set_title('left mic outlier')
        ax2 = plt.subplot(212)
        ax2.plot(time,echo[j[0],0,:])
        ax2.hlines(y=thres,xmin=10,xmax=25,color='r',linestyle='--')
        ax2.hlines(y=-thres,xmin=10,xmax=25,color='r',linestyle='--')
        ax2.set_ylim((-1.2*thres,1.2*thres))
        ax2.set_xlabel("Time [ms]")
        ax2.set_ylabel("Amplitude")
        plt.show()
    
for i in right_idx:
    for j in i:
        print('right mic')
        print(j)
        ax1 = plt.subplot(211)
        ax1.plot(time,echo[j[0],1,:])
        ax1.hlines(y=thres,xmin=10,xmax=25,color='r',linestyle='--')
        ax1.hlines(y=-thres,xmin=10,xmax=25,color='r',linestyle='--')
        ax1.set_ylim((-0.5,0.5))
        ax1.set_ylabel("Amplitude")
        ax1.set_title('right mic outlier')
        ax2 = plt.subplot(212)
        ax2.plot(time,echo[j[0],1,:])
        ax2.hlines(y=thres,xmin=10,xmax=25,color='r',linestyle='--')
        ax2.hlines(y=-thres,xmin=10,xmax=25,color='r',linestyle='--')
        ax2.set_ylim((-1.2*thres,1.2*thres))
        ax2.set_xlabel("Time [ms]")
        ax2.set_ylabel("Amplitude")
        plt.show()  
    
    
    
    
