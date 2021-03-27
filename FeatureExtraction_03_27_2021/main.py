from build_dataset import build_dataset
from pyimagesearch import config
from extract_features import extract_features
from train import train
import numpy as np
import shutil
import os
import time

s = time.time()
print(config.BASE)
os.chdir(config.BASE)
image_list = os.listdir()
image_list = [str(i) for i in image_list if '.jpg' in str(i)]
total = len(image_list)

if not os.path.exists(config.ORIG_INPUT_DATASET):
    os.makedirs(config.ORIG_INPUT_DATASET)
if not os.path.exists("dataset"):
    os.makedirs("dataset")
if not os.path.exists(config.ORIG_INPUT_DATASET + "\\evaluation"):
    os.makedirs(config.ORIG_INPUT_DATASET + "\\evaluation")
if not os.path.exists(config.ORIG_INPUT_DATASET + "\\training"):
    os.makedirs(config.ORIG_INPUT_DATASET + "\\training")
if not os.path.exists("output"):
    os.makedirs("output")

x = np.random.choice(range(total), total, replace=False)
e = int(total / 5)

for i in x[:e]:
    shutil.move(image_list[i], config.ORIG_INPUT_DATASET + "\\evaluation\\" + image_list[i])
for i in x[e:]:
    shutil.move(image_list[i], config.ORIG_INPUT_DATASET + "\\training\\" + image_list[i])

build_dataset()
extract_features()
# train(info = "Result of Right Mic with Feature Extraction - VGG\n Random choose 500 gap 500 foliage for all 10 datasets \n Revised each of the spectrogram \n \t 1. with half of overlap. \n \t2. with half of overlap and half of window length\n Total amount of images 10000")
train(
    info="Result of Right Mic with Feature Extraction - VGG\nRandom choose 500 gap 500 foliage for all 10 datasets \nRevised each of the spectrogram\n\twindow length: longest, overlap: 0\nTotal amount of images 9400")



e = time.time()
print("Duration: " +  str(e-s))
