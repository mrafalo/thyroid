import os
os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]
import tensorflow as tf

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import work.models as m
import work.data as d
import work
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
import yaml    
import logging
from datetime import datetime
import utils
import random
from datetime import timedelta
import importlib
import timeit
from vit_keras import vit, utils, visualize


with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    BASE_PATH = cfg['BASE_PATH']
    IMG_WIDTH = cfg['IMG_WIDTH']
    IMG_HEIGHT = cfg['IMG_HEIGHT']


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

TRAIN_PATH = BASE_PATH + 'modeling/all_images/train/'
VAL_PATH = BASE_PATH + 'modeling/all_images/validation/'
TEST_PATH = BASE_PATH + 'modeling/all_images/test/'

TRAIN_PATH_BW = BASE_PATH + 'modeling/all_images_bw/train/'
TEST_PATH_BW = BASE_PATH + 'modeling/all_images_bw/test/'
VAL_PATH_BW = BASE_PATH + 'modeling/all_images/bw/validation/'

TRAIN_PATH_SOBEL = BASE_PATH + 'modeling/all_images_sobel/train/'
TEST_PATH_SOBEL = BASE_PATH + 'modeling/all_images_sobel/test/'
VAL_PATH_SOBEL = BASE_PATH + 'modeling/all_images/sobel/validation/'

TRAIN_PATH_HEAT = BASE_PATH + 'modeling/all_images_heat/train/'
TEST_PATH_HEAT = BASE_PATH + 'modeling/all_images_heat/test/'
VAL_PATH_HEAT = BASE_PATH + 'modeling/all_images/heat/validation/'

TRAIN_PATH_CANNY = BASE_PATH + 'modeling/all_images_canny/train/'
TEST_PATH_CANNY = BASE_PATH + 'modeling/all_images_canny/test/'
VAL_PATH_CANNY = BASE_PATH + 'modeling/all_images/canny/validation/'

TRAIN_PATH_FELZEN = BASE_PATH + 'modeling/all_images_felzen/train/'
TEST_PATH_FELZEN = BASE_PATH + 'modeling/all_images_felzen/test/'
VAL_PATH_FELZEN = BASE_PATH + 'modeling/all_images/felzen/validation/'

INPUT_PATH = BASE_PATH + 'modeling/all_images/'
OUTPUT_TEST_PATH = TEST_PATH
OUTPUT_TRAIN_PATH = TRAIN_PATH
OUTPUT_VAL_PATH = VAL_PATH
   


# Load a model
image_size = 384
classes = utils.get_imagenet_classes()
model = vit.vit_b16(
    image_size=image_size,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)
classes = utils.get_imagenet_classes()

# Get an image and compute the attention map
url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
image = utils.read(url, image_size)
attention_map = visualize.attention_map(model=model, image=image)
print('Prediction:', classes[
    model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
)  # Prediction: Eskimo dog, husky

# Plot results
fig, (ax1, ax2) = plt.subplots(ncols=2)
ax1.axis('off')
ax2.axis('off')
ax1.set_title('Original')
ax2.set_title('Attention Map')
_ = ax1.imshow(image)
_ = ax2.imshow(attention_map)    