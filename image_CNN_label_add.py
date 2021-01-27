# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:45:46 2021
Before run this code, make sure you have picked up all frames where LED is on,
and save them as 'image_total.npy'.
Function:
    1). Load the 'image_total.npy' from destination folder.
    2). Randomly pick up N images as lottery, and mannual label them as f(foliage), g(gap), n(unclear)).
    3). Save the lottery and labelled images. Get ready to train them on CNN.
@author: wrhsd
"""
import os
import numpy as np
from window_label import window_label

# Here is the destination folder name
s = 'Outdoor_6_a'
dirpath = "C:\\Users\\wrhsd\\OneDrive\\Desktop\\Run\\Data\\Outdoor\\Image\\%s\\" % s
os.chdir(dirpath)
# Load the 'image_total.npy'
image_total = np.load('image_total.npy')
label, label_check = [],[]
# define the lottery parameter. L is the number of frames, size is the lottery size (we choose 500).  
L = image_total.shape
lottery = np.random.randint(L[0],size=500)
# Use window_label function to label the image.
label = np.append(label,[window_label(image_total[i]) for i in lottery])
np.save('lottery.npy',lottery)
np.save('label.npy',label)
# During the labelling, we may input wrong character (none of 'f','g' or 'n')
# then we need to find these wrong input and re-label them.
idx_nf = np.where(label!='f')
idx_ng = np.where(label!='g')
idx_nn = np.where(label!='n')
idx_check = np.intersect1d(idx_nn, np.intersect1d(idx_nf,idx_ng))
label_check = np.append(label_check,[window_label(image_total[lottery[i]]) for i in idx_check])
label[idx_check] = label_check
# Save the final label
np.save('label.npy',label)


if __name__ == '__main__':
    print("Number of images: %d, %d foliage, %d gap, %d unclear" 
          % (len(label),len(np.where(label=='f')[0]),len(np.where(label=='g')[0]),
             len(np.where(label=='n')[0])))