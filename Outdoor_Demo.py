# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:31:01 2020
This code is used to show the echo in time domain, spectrogram and the 
image when pulse sent. 
@author: Ruihao Wang
"""
import os
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, show, draw, pause, subplot, xlabel, ylabel, title
import matplotlib.image as img
import matplotlib.patches as pat
from pathlib import Path
from operator import itemgetter
import cv2


# Go to dir
s = 'Outdoor_9_a'
path_echo = "D:\\RW\\Sonar_Echo\\Raw_Data\\Outdoor\\Echo\\%s\\" % s
path_image = "D:\\RW\\Sonar_Echo\\Raw_Data\\Outdoor\\Image\\%s\\" % s
os.chdir(path_echo)
# bandpass filter parameter
lowcut = 15e3
highcut = 150e3
# sigal parameter (fs: sampling frequency, n_sample: number of sample)
fs = 400e3
n_sample = 10000
time = np.linspace(0,1000*(n_sample/fs),n_sample)
distance = time * 340 / 2 *(1/1000)

"""
Define all function
"""
# sort the echo files by time
def sort_echo_path(path_echo):
    path = []
    path = [i for i in os.listdir(path_echo) if '.txt' in i]
    num = [int(i.split('-')[1]) for i in os.listdir(path_echo) if '.txt' in i]
    sort_idx = sorted(range(len(num)), key=lambda k: num[k])
    sort_path = [path[i] for i in sort_idx]
    return sort_path

# plot echo with time/distance as x-axis, just for observation.
def echo_plt(data,time,distance,i):
    fig = plt.figure()
    f,(ax1,ax2) = plt.subplots(2,1,sharey=True)
    ax12 = ax1.twiny()
    ax22 = ax2.twiny()
    # ax1.plot(time,data[0])
    ax12.plot(distance,data[0])
    ax2.plot(time,data[1])
    # ax22.plot(distance,data[1])
    ax12.cla
    ax22.cla
    # ax1.set_xlabel("Time [ms]")
    ax1.set_xticks([])
    ax2.set_xlabel("Time [ms]")
    ax12.set_xlabel("Distance [m]")
    # ax22.set_xlabel("Distance [m]")
    ax22.set_xticks([])
    ax1.set_ylabel("Amplitude (Left)")
    ax2.set_ylabel("Amplttude (Right)")
    plt.show()
    # f.savefig("%d_echo.png" % i)
    
# plot spectrogram, really ugly figure
def spectrogram(x,fs,nfft,overlap,i):
    fig = plt.figure()
    ax3 = subplot(2,1,1)           
    ax3.specgram(x[0],Fs=fs,NFFT=nfft,noverlap=overlap,cmap='jet')
    ax3.set_ylim(15,150)
    ax3.set_ylabel("Frequency [kHz] (Left)")
    ax4 = subplot(2,1,2)
    ax4.specgram(x[1],Fs=fs,NFFT=nfft,noverlap=overlap,cmap='jet')
    ax4.set_ylim(15,150)
    ax4.set_ylabel("Frequency [kHz] (Right)")
    ax4.set_xlabel("Time [ms]")
    # plt.tight_layout()
    plt.show()
    # fig.savefig("%d_spectrogram.png" % i)

# bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# read txt file from folder, then bandpass, then normalize, then remove DC shift
def read_data(file):
    A = np.loadtxt(file)
    if len(A) == 20000:
        echo1 = A[0:200002:2]
        echo2 = A[1:200002:2]
        echo = (echo1,echo2)
        filtecho = butter_bandpass_filter(echo,lowcut,highcut,fs,5)
        normecho = (filtecho-np.min(filtecho))/(np.max(filtecho)-np.min(filtecho))
        aveecho = normecho - np.mean(normecho)
    else:
        aveecho = np.zeros((2,10000))
    return aveecho

# save echo into numpy array
def prep_echo(echo):
    count = len(echo)
    data = np.ndarray((count,2,10000), dtype=np.float)
    for i, echo_file in enumerate(echo):
        data[i] = read_data(echo_file)
        if i%50 == 0:
            print('Processed {} of {}'.format(i, count))
    return data

def main():
    raw_echo = sort_echo_path(path_echo)
    filted = prep_echo(raw_echo)
    np.save('filted_echo.npy',filted)
    # os.chdir(path_image)
    # image = np.load('image_tri.npy')
    # echo_plt(filted[10],time,distance)
    # spectrogram(filted[10],fs/1e3,256,200)
    # plt.imshow(image[10])
    # plt.show()
    # for i in range(len(raw_echo)):
    #     echo_plt(filted[i],time,distance,i)
    #     spectrogram(filted[i],fs/1e3,256,200,i)
    #     fig = plt.figure()
    #     plt.imshow(image[i])
    #     plt.show()
    #     fig.savefig('%d_camera.png' % i)
        

if __name__ == '__main__':
    main()
    
    
# for i in range(0,3634,200):
#     resized = cv2.resize(image[i,200:,:],(1200,600),interpolation = cv2.INTER_AREA)
#     fig = plt.figure()
#     ax = fig.add_subplot(111) 
#     plt.imshow(np.fliplr(resized),cmap='gray',vmin=0,vmax=255)
#     cir1 = pat.Circle((600,250),333/2,fc='none',ec='r',ls='-')
#     cir2 = pat.Circle((600,250),170/2,fc='none',ec='r',ls='-.')
#     cir3 = pat.Circle((600,250),117/2,fc='none',ec='r',ls=':')
#     ax.add_patch(cir1)
#     ax.add_patch(cir2)
#     ax.add_patch(cir3)
#     # plt.axis('off')
#     plt.autoscale(tight=True)
#     plt.show()
#     echo_plt(echo[i,:,:], time, distance,i)
#     spectrogram(echo[i,:,:],fs/1e3,256,200,i)
#     # fig.savefig('%d_camera.png' % i)
#     # plt.close()

# num = 100
# input = filted[num]
# echo_plt(input, time,distance,1)
# spectrogram(input,fs/1e3,256,200,1)
    
    