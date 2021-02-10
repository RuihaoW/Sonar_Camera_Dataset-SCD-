from pyimagesearch import config
import os
import numpy as np
from shutil import copyfile
from pathlib import Path

f = open(config.BASE_CSV_PATH + '\\image_name.txt','r').read().split('\n')
label = np.load(config.BASE_CSV_PATH + '\\label.npy')


os.makedirs(config.BASE_CSV_PATH + '\\foliage')
os.makedirs(config.BASE_CSV_PATH + '\\gap')

for i in range(len(label)):
	if (label[i] == "1"):
		copyfile(f[i], config.BASE_CSV_PATH + '\\gap\\1_' + str(i) +'.jpg' )
	else:
		copyfile(f[i], config.BASE_CSV_PATH + '\\foliage\\0_' + str(i) +'.jpg' )