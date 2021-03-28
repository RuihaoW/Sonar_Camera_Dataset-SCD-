import os
import numpy as np

os.chdir("F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\test")
dataset_name = ['2_a','4_b','6_a','7_b','9_a','9_d_1','9_d_2','10_a','12_a','13_c']
image_list = os.listdir()
index = []
deliminator = "_"

index.append(np.int32(-1))

for d in dataset_name:
	for i in image_list:
		i = i.split("_")
		if deliminator.join(i[:-3][1:]) == d:
			index.append(np.int32(i[:-2][-1]))
	index.append(np.int32(-1))
np.save("index.npy", index)

