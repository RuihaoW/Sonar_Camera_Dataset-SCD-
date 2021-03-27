# import the necessary packages
import os
 
# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "ForG"
 
# initialize the base path to the *new* directory that will contain
# # # our images after computing the training and testing split\
# BASE = "F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\original\\original_right"
BASE = "F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\right_wl_4\\"
BASE_PATH = BASE + "\\dataset"
# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"
 
# initialize the list of class label names
CLASSES = ["gap", "foliage"]
 
# set the batch size
BATCH_SIZE = 16

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"
 
# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "model.cpickle"])