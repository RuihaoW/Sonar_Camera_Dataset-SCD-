# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:42:22 2020
This code is used to pre-process the video/image data from GoPro
This is the second version of image_preprocess, differnt from first version,
this version will foucos on LED area only and R and G channels.
1. Select the image where LED is on
2. Save the selected images into array.
3. This code only return index of image where the LED is on.
@author: Ruihao Wang
"""
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import  find_peaks, medfilt
from scipy.signal.windows import tukey
import time
# from numba import vectorize, jit, cuda

# Go to dir
s = 'Outdoor_9_b'
dirpath = "C:\\Users\\wrhsd\\OneDrive\\Desktop\\Run\\Data\\Outdoor\\Image\\%s\\Raw" % s
os.chdir(dirpath)
path = sorted(Path(dirpath).iterdir(), key=os.path.getctime)
# we want to resize the images to ROW, COL
ROW, COL = 20,15
# tri save the images idx where LED is on
tri = []
"""
Define all functions
"""
        
# Read images from path, change color style, resize/flip/rotate.
def read_LED(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #_COLOR/_GRAYSCALE
    cropped = img[695:715,510:525,:]
    b,g,r = cv2.split(cropped)
    return r,g

# Save images data into array
def prep_LED(images):
    count = len(images)
    data_r = np.ndarray((count,ROW,COL), dtype=np.float32)
    data_g = np.ndarray((count,ROW,COL), dtype=np.float32)
    for i, image_file in enumerate(images):
        data_r[i],data_g[i] = read_LED(image_file)
        if i%1000 == 0:
            print('Processed {} of {}'.format(i, count))
    return data_r,data_g

"""
Read raw images
"""
# raw_images =   [path+i for i in os.listdir(str(path)) if '.jpg' in i]
raw_images =  [str(i) for i in path if '.jpg' in str(i)]
"""
Test
"""
test = raw_images[15035]
img = cv2.imread(test,cv2.IMREAD_COLOR)
plt.imshow(img)
plt.imshow(img[695:715,510:525,:])

t = time.time()
led_R,led_G = prep_LED(raw_images[:-1])
r,g = np.mean(led_R,axis=(1,2)), np.mean(led_G,axis=(1,2))
elapsed = np.floor((time.time()-t)/60)
print('The time cost is {} minutes'.format(elapsed))
print(np.size(led_R),np.size(led_G))

# median filter
k_size = 19
r_med, g_med = medfilt(r,kernel_size = k_size), medfilt(g, kernel_size= k_size)  
#  tunkey window filter
alpha,n = 0.5,15
sq_win = tukey(n,alpha=alpha)
r_in = r-r_med + abs(g-g_med)
w = []
for i in range(len(r)-n):
    w = np.append(w,r_in[i:i+n]@sq_win)
xlim1,xlim2 = 2e4,2.02e4
plt.plot(w)
plt.xlim(xlim1,xlim2)
plt.title(alpha+n)


pks = find_peaks(w[1200:],height=20,distance=10) 
p = pks[0]+1200 

I = w[p]
I_sort = np.sort(I)
I_sort_idx = np.argsort(I)
# fft
# N = len(w)
# T = 1.0 / 60
# yf = fft(w)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()

# os.chdir('..')

ax1 = plt.subplot(311)
ax1.plot(r,color='red')
ax1.plot(r_med)
ax1.plot(r-r_med)
ax1.set_xlim(xlim1,xlim2)
ax1.legend(['r','med','r-med'])
ax1.set_title(k_size)

ax2 = plt.subplot(312)
ax2.plot(g,color='green')
ax2.plot(g_med)
ax2.plot(g-g_med)
ax2.set_xlim(xlim1,xlim2)
ax2.legend(['g','g','g-med'])

ax3 = plt.subplot(313)
ax3.plot(r-r_med + abs(g-g_med))
ax3.plot(w)
ax3.scatter(p,I)
ax3.set_xlim(xlim1,xlim2)
ax3.legend(['r and g'])

plt.tight_layout()
plt.show()

p = p + np.floor(n/2)
image_idx = p.astype(int)
np.save('images_idx.npy',image_idx)
