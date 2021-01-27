# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:02:01 2020

@author: wrhsd
"""

import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from math import floor as floor
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import  Callback, EarlyStopping
# Define parameters, load images_idx.npy (Index of images where LED is on).
s = 'Outdoor_9_b'
dirpath = "C:\\Users\\wrhsd\\OneDrive\\Desktop\\Run\\Data\\Outdoor\\Image\\%s" % s
os.chdir(dirpath)
idx = np.load('images_idx.npy')
label = np.load('label.npy')
beamwidth = np.load('beam_width.npy')
dirpath = "C:\\Users\\wrhsd\\OneDrive\\Desktop\\Run\\Data\\Outdoor\\Image\\%s\\Raw\\" % s
os.chdir(dirpath)
path = sorted(Path(dirpath).iterdir(), key=os.path.getctime)
# Set array size for imges. ROWS and COlS are length and width, 
# Channels is layers, 1 is for gray, 3 is for color
ROWS = beamwidth
COLS = beamwidth
CHANNELS = 3

# Function read_image is to read all .jpg images, resize them into square shape
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_COLOR
    img = cv2.flip(img,0)
    img = img[floor((720-beamwidth)/2):floor((720+beamwidth)/2), 
              floor((1280-beamwidth)/2):floor((1280+beamwidth)/2),:]
    # resized = np.expand_dims(resized,axis = 2)
    return img
# Function prep_data is to convert images into array.
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i%1000 == 0:
            print('Processed {} of {}'.format(i, count))
    return data
print('Preparing training data')
raw_images = [str(i) for i in path if '.jpg' in str(i)]
idx_images = [raw_images[i] for i in idx]
total = prep_data(idx_images)
print("Dataset shape: {}".format(total.shape))
dirpath = "C:\\Users\\wrhsd\\OneDrive\\Desktop\\Run\\Data\\Outdoor\\Image\\%s" % s
os.chdir(dirpath)
np.save('image_total.npy',total)
# Label the images by 1 (for dog) and 0 (for cat)
labels = []
for i in label:
    if 'g' in i:
        labels.append(0)
    else:
        labels.append(1)
print("Total labels: %d foliage and %d gap" %(len((np.where(np.asarray(labels)==1))[0]), 
                                              len((np.where(np.asarray(labels)==0))[0])))

X_train, X_test, y_train, y_test = train_test_split(total,labels,test_size=0.20)

#  Setup the parameter to update weights and bias in the CNN model
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'
#  Function catdog() is a 12-layers CNN model, containing 9 convolutional layers
#  and 3 fully connected layers.
def catdog():
    model = Sequential()
    model.add(Conv2D(16, 3, padding='same', input_shape= (ROWS,COLS,CHANNELS), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(16, 3,padding = 'same',activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))   
    model.add(Conv2D(32, 3,padding = 'same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    # model.add(Conv2D(32, 3,padding = 'same',activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))   
    # model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, 3, padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    # Here is the first dropout function
    model.add(Dropout(0.5))    
    model.add(Dense(256, activation='relu'))
    # # Here is the second dropout function
    model.add(Dropout(0.5))    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))    
    print("Compiling model...")
    model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])
    return model
print("Creating model:")
model = catdog()
# Set epochs and batch size for training
epochs = 100
batch_size = 128
# Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []      
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
early_stopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
# function run_catdog() is to train models. It divides the training data into 75% and 25%.
# 75% for training, 25% for validation.
def run_catdog():  
    history = LossHistory()
    print("running model...")
    model.fit(X_train, np.asarray(y_train), batch_size=batch_size, epochs=epochs,
              validation_data=(X_test, np.asarray(y_test)), verbose=2, shuffle=True, callbacks=[history,early_stopping])    
    return history
history = run_catdog()
loss = history.losses
val_loss = history.val_losses
# Plot the loss in training and validation data and compare them in one figure
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Foliage-Gap Image Classification Loss Trend')
plt.ylim(0,1)
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,epochs)[0::2])
plt.legend()
plt.show()

model.save('model_weight.h5')
model.save_weights('model.h5')

prediction = model.predict(X_test)

for i in range(0, len(y_test),30):
    plt.imshow(X_test[i,:,:,:])
    if y_test[i] == 1:
        l = 'The label is foliage'
        if prediction[i] >0.5:
            p = ", the prediction is foliage({:0.2f}%)".format(100*float(prediction[i]))
        else:
            p = ", the prediction is gap({:0.2f}%)".format(100*float(1-prediction[i]))
        plt.title(l+p)
    else:
        l = 'The label is gap'
        if prediction[i] >0.5:
            p = ", the prediction is foliage({:0.2f}%)".format(100*float(prediction[i]))
        else:
            p = ", the prediction is gap({:0.2f}%)".format(100*float(1-prediction[i]))
        plt.title(l+p)
    plt.show()

TP,TN,FP,FN = 0,0,0,0    
for i in range(len(y_test)):
    if prediction[i]> 0.5 and y_test[i] == 1:
        TP = TP + 1
    elif prediction[i]>0.5 and y_test[i] == 0:
        FP = FP + 1
        print(i)
    elif prediction[i]<0.5 and y_test[i] == 0:
        TN = TN + 1
    elif prediction[i]<0.5 and y_test[i] == 1:
        FN = FN + 1
        print(i)
        
acc = (TP+TN)/(TP+TN+FP+FN)
    


for i in FP_list:
    plt.imshow(X_test[i,:,:,:])
    if y_test[i] == 1:
        l = 'The label is foliage'
        if prediction[i] >0.5:
            p = ", the prediction is foliage({:0.2f}%)".format(100*float(prediction[i]))
        else:
            p = ", the prediction is gap({:0.2f}%)".format(100*float(1-prediction[i]))
        plt.title(l+p)
    else:
        l = 'The label is gap'
        if prediction[i] >0.5:
            p = ", the prediction is foliage({:0.2f}%)".format(100*float(prediction[i]))
        else:
            p = ", the prediction is gap({:0.2f}%)".format(100*float(1-prediction[i]))
        plt.title(l+p)
    plt.show()

    

