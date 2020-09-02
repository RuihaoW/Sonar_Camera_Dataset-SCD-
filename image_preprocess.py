# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:29:54 2020
This code is used to pre-process the video/image data from GoPro
1. Select the image where LED is on
2. Save the selected images into array.
@author: Ruihao Wang
"""
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.io import savemat




# Go to dir
s = 'Outdoor_3_a'
dirpath = "D:\\RW\\Sonar_Echo\\Raw_Data\\Outdoor\\Image\\%s\\Raw\\" % s
os.chdir(dirpath)
path = sorted(Path(dirpath).iterdir(), key=os.path.getmtime)
# we want to resize the images to ROW, COL
ROW, COL = 1280,720
# tri save the images idx where LED is on
tri = []
tri_up = []
tri_dn = []
tri_new = []
"""
Define all functions
"""
        
# Read images from path, change color style, resize/flip/rotate.
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) #_COLOR/_GRAYSCALE
    # resized = cv2.resize(img, (ROW, COL), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(img, (ROW, COL), interpolation=cv2.INTER_CUBIC)
    resized = np.expand_dims(resized,axis = 2)
    resized = np.flipud(resized)
    return resized

# Save images data into array
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, COL, ROW,1), dtype=np.float32)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i%100 == 0:
            print('Processed {} of {}'.format(i, count))
    return data

"""
Read raw images
"""
# raw_images =   [path+i for i in os.listdir(str(path)) if 'jpg' in i]
raw_images =   [str(i) for i in path if 'jpg' in str(i)]
dur = 1000
times = len(raw_images)//dur
for i in range(times):
    # i = 1
    resized = []
    resized = prep_data(raw_images[i*dur:(i+1)*dur-1])
    # Commands to imshow/pcolor to show image
    # num =1965
    # plt.pcolor(resized[num,0:25,600:620,0],vmin=0,vmax=70)
    # plt.title(num)
    # plt.colorbar()
    # plt.plot()
    # Foucs on the LED area to measure on/off
    # 20:70,575:620 for full size image; 2:5,58:60 for small resized image.
    LED_area = resized[:,5:25,510:525,0] 
    L = np.mean(LED_area,axis = (1,2))
    L = np.square(L)
    # S1: by peak
    # peaks,_ = find_peaks(L,distance = 15)
    # tri = np.append(tri,peaks+i*dur)
    # S2: by max value
    # tri = np.append(tri, [i for i,arg in enumerate(L) if arg>60 and L[i]==max(L[i-5:i+5])])
    # tri = np.delete(tri,[i for i in range(len(tri)-1) if (tri[i+1]-tri[i])<5])
    # tri = tri.astype(int)
    # S3: by up step and down step
    for j in range(dur-2):
        if L[j+1]-L[j] > 500:
            tri_up = np.append(tri_up,j+i*dur)
        elif L[j+1]-L[j] < -500:
            tri_dn = np.append(tri_dn,j+i*dur)
    tri_new = np.append(tri_new, np.asarray(np.where(L > 600)) + i * dur)
            
tri_up= np.delete(tri_up,[i for i in range(len(tri_up)-1) if (tri_up[i+1]-tri_up[i])<5])
tri_dn= np.delete(tri_dn,[i for i in range(len(tri_dn)-1) if (tri_dn[i+1]-tri_dn[i])<5])    
tri = tri_up[1:]+1
tri = tri.astype(int)

tri_new_2 = np.delete(tri_new,[i for i in range(len(tri_new)-1) if (tri_new[i+1]-tri_new[i])<5])

resized = []
resized = prep_data([raw_images[i] for i in tri])                 

    
os.chdir('..')
np.save('image_tri.npy', resized[:,:,:,0])
np.save('image_tri_1.npy',resized[:10,:,:,0])


# simple test
# plt.plot(tri_dn,'blue')
# plt.plot(tri_up,'green')
# plt.show()

# d = []
# for i in range(1525-1):
#     d.append(tri[i+1]-tri[i])

# plt.plot(d)
# plt.ylim([0,50])


