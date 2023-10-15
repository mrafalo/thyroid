import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np

import importlib
import work
import work.models as m
import work.data as d
import utils
import utils.image_manipulator as im
import tensorflow as tf
from keras import backend as K

from matplotlib import pyplot
from tensorflow.keras.optimizers import RMSprop, Adam
import yaml    
import logging
import keras
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from scipy.stats import chi2_contingency
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn import tree
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    BASE_PATH = cfg['BASE_PATH']
    IMG_WIDTH = cfg['IMG_WIDTH']
    IMG_HEIGHT = cfg['IMG_HEIGHT']


def get_layers_info(_model):
    print("input layer:", _model.input_names)
    
    for layer in _model.layers:
        if 'conv' not in layer.name:
            print(layer.name)
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)
    
    print("output layer:", m1.output_names)


def plot_kernels(_model):
    
    filters, biases = _model.layers[2].get_weights()
    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    # plot first few filters
    n_filters, ix = 6, 1
    for i in range(n_filters):
    	# get the filter
    	f = filters[:, :, :, i]
    	# plot each channel separately
    	for j in range(3):
    		# specify subplot and turn of axis
    		ax = pyplot.subplot(n_filters, 3, ix)
    		ax.set_xticks([])
    		ax.set_yticks([])
    		# plot filter channel in grayscale
    		pyplot.imshow(f[:, :, j], cmap='gray')
    		ix += 1
    # show the figure
    pyplot.show()

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s(%(name)s) %(levelname)s: %(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
if (logger.hasHandlers()):
    logger.handlers.clear()
  
logger.addHandler(ch)

BASE_FILE_PATH = BASE_PATH + 'baza6.csv'



# m1 = train_and_save(30, 'models/m1')
m1 = keras.models.load_model('models/m1')
imp_path = "C:/datasets/COI/v2/baza/modeling/all_images/test/base_resized_out_rec_from_shape_242a_1799_12.png"
val_file = d.img_to_predict("C:/datasets/COI/v2/baza/modeling/all_images/test/base_resized_out_rec_from_shape_242a_1799_12.png")

y_pred = m1.predict(val_file)
print(y_pred)

get_layers_info(m1)

plot_kernels(m1)


m1.summary()

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=m1.layers[0].output)
model.summary()
# load the image with the required shape
img = load_img(imp_path)

img = np.array(cv2.imread(imp_path,cv2.IMREAD_GRAYSCALE))

img.shape

img= img.reshape(1,180,180,1)
# convert the image to an array
#img = img_to_array(img)
img.shape


# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
img.shape
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
img.shape

feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
square = 3
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()