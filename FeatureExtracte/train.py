# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from pyimagesearch import config
import numpy as np
import pickle
import os


def load_data_split(splitPath):
	# initialize the data and labels
	data = []
	labels = []
 
	# loop over the rows in the data split file
	for row in open(splitPath):

		# extract the class label and features from the row
		row = row.strip().split(",")
		label = row[0]
		features = np.array(row[1:], dtype="float")
 
		# update the data and label lists
		data.append(features)
		labels.append(label)
 
	# convert the data and labels to NumPy arrays
	data = np.array(data)
	labels = np.array(labels)
 
	# return a tuple of the data and labels
	return (data, labels)


# derive the paths to the training and testing CSV files
trainingPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TRAIN)])
testingPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.TEST)])
validationPath = os.path.sep.join([config.BASE_CSV_PATH,
	"{}.csv".format(config.VAL)])

# load the data from disk
print("[INFO] loading data...")
(trainX, trainY) = load_data_split(trainingPath)
(testX, testY) = load_data_split(testingPath)
(valX, valY) = load_data_split(validationPath)

# load the label encoder from disk
le = pickle.loads(open(config.LE_PATH, "rb").read())

# train the model
print("[INFO] training model...")
# LR
model = LogisticRegression(solver="lbfgs", multi_class="auto")
model.fit(trainX, trainY)
# LDA
model = LinearDiscriminantAnalysis()
model.fit(trainX, trainY)
# SVM
model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
model.fit(trainX, trainY)
 
# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(testX)
val = model.predict(valX)


print(classification_report(testY, preds, target_names=le.classes_))
print(classification_report(valY, val, target_names=le.classes_))

# serialize the model to disk
print("[INFO] saving model...")

np.save(config.BASE_CSV_PATH + '\\label.npy',val)
np.save(config.BASE_CSV_PATH+ '\\picture.npy', valX)
with open(config.MODEL_PATH, "wb") as f:
    f.write(pickle.dumps(model))