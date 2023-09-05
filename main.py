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
from tensorflow.keras.optimizers import RMSprop, Adam
import yaml    
import logging
import keras
from matplotlib import pyplot as plt
import random
import wandb
import copy
from wandb.keras import WandbMetricsLogger
from datetime import timedelta

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
import timeit

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

 
BASE_FILE_PATH = BASE_PATH + 'baza5.csv'

TRAIN_PATH = BASE_PATH + 'modeling/all_images/train/'
VAL_PATH = BASE_PATH + 'modeling/all_images/validation/'
TEST_PATH = BASE_PATH + 'modeling/all_images/test/'

TRAIN_PATH_BW = BASE_PATH + 'modeling/all_images_bw/train/'
TEST_PATH_BW = BASE_PATH + 'modeling/all_images_bw/test/'

TRAIN_PATH_SOBEL = BASE_PATH + 'modeling/all_images_sobel/train/'
TEST_PATH_SOBEL = BASE_PATH + 'modeling/all_images_sobel/test/'

TRAIN_PATH_HEAT = BASE_PATH + 'modeling/all_images_heat/train/'
TEST_PATH_HEAT = BASE_PATH + 'modeling/all_images_heat/test/'

TRAIN_PATH_CANNY = BASE_PATH + 'modeling/all_images_canny/train/'
TEST_PATH_CANNY = BASE_PATH + 'modeling/all_images_canny/test/'

TRAIN_PATH_FELZEN = BASE_PATH + 'modeling/all_images_felzen/train/'
TEST_PATH_FELZEN = BASE_PATH + 'modeling/all_images_felzen/test/'

BATCH_SIZE = 12
EPOCHS = 100
BASE_LR = 0.001#1e-6
VERBOSE_FLAG = False


def train_model(_x_train, _y_train, _x_test, _y_test, _epochs):
    
    m1 = m.model_cnn1(IMG_WIDTH, IMG_HEIGHT)
    print("processing started...")
    print("cancer ratio in train data is:", round(sum(_y_train[:,1])/len(_y_train[:,1]),2))
    print("cancer ratio in test data is:", round(sum(_y_test[:,1])/len(_y_test[:,1]),2))
    m1.compile(optimizer = RMSprop(learning_rate=BASE_LR), loss='categorical_crossentropy', metrics=["accuracy"]) 
    hist = m1.fit(_x_train, _y_train, validation_data=(_x_test, _y_test), batch_size=BATCH_SIZE, epochs=_epochs)
    
    ev = m1.evaluate(_x_test, _y_test)
    
    print("processing finished!");
    return ev
     

def reinitialize(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
            
def get_config():
    res = pd.DataFrame(columns = ['learning_rate', 'batch_size', 'optimizer'])
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [8, 16, 32]
    optimizers = ['Adam', 'SDG']
    
    for l in learning_rates:
        for b in batch_sizes:
            for o in optimizers:        
                new_row = {'learning_rate':l, 'batch_size':b, 'optimizer':o}
                res = res.append(new_row, ignore_index=True)
                
    return res

            
def train_model_multi_cv(_epochs, _iters, _filter="none", _cancer_filter='none'):
    
    
    if _filter == "none": 
        INPUT_PATH = BASE_PATH + 'modeling/all_images/'
        OUTPUT_TEST_PATH = TEST_PATH
        OUTPUT_TRAIN_PATH = TRAIN_PATH
        OUTPUT_VAL_PATH = VAL_PATH
    if _filter == "canny":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_canny/'
        OUTPUT_TEST_PATH = TEST_PATH_CANNY
        OUTPUT_TRAIN_PATH = TRAIN_PATH_CANNY
    if _filter == "heat":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_heat/'
        OUTPUT_TEST_PATH = TEST_PATH_HEAT
        OUTPUT_TRAIN_PATH = TRAIN_PATH_HEAT
    if _filter == "sobel":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_sobel/'
        OUTPUT_TEST_PATH = TEST_PATH_SOBEL
        OUTPUT_TRAIN_PATH = TRAIN_PATH_SOBEL
    if _filter == "bw":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_bw/'
        OUTPUT_TEST_PATH = TEST_PATH_BW
        OUTPUT_TRAIN_PATH = TRAIN_PATH_BW
    if _filter == "felzen":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_felzen/'
        OUTPUT_TEST_PATH = TEST_PATH_FELZEN
        OUTPUT_TRAIN_PATH = TRAIN_PATH_FELZEN

    
    models = m.model_sequence_manual_3(IMG_WIDTH, IMG_HEIGHT)
    model_cnt = len(models)
    
    logger.info('processing start... ' + 'models: ' + str(model_cnt) + ' cv iters: ' + str(_iters) + ' filter: ' + str(_filter))
    
    config = get_config()
            
    for i in range(_iters):
        
        d.split_files(INPUT_PATH, OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH, OUTPUT_TEST_PATH, 0.15, 0.15)
                
        X_train, y_train, X_val, y_val, X_test, y_test = d.split_data(BASE_FILE_PATH, OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH, OUTPUT_TEST_PATH, 0, _cancer_filter)
        
        for idx, c in config.iterrows():
            
            for m_num in range(model_cnt):
                
                start = timeit.default_timer()
                models = m.model_sequence_manual_3(IMG_WIDTH, IMG_HEIGHT)
                m1 = models[m_num]
                
                keras.backend.clear_session()
                
                ev = m.model_fitter(m1, X_train, y_train, X_val, y_val, X_test, y_test, _epochs, c['learning_rate'], c['batch_size'], c['optimizer']);
                            
                
                stop = timeit.default_timer()

                elapsed = timedelta(minutes=stop-start)

                histories = pd.DataFrame(columns =["model_num", "iter_num", "cancer_filter", "filter",  "cancer_ratio_train", 
                                                   "cancer_ratio_test","accuracy", "train_dataset_size", "test_dataset_size",
                                                   "learning_rate", "batch_size", "optimizer", "elapsed"])
                new_row = {'model_num':m_num+1,
                           'iter_num':i+1,
                           'cancer_filter':_cancer_filter, 
                           'filter':_filter, 
                           'cancer_ratio_train':round(sum(y_train[:,1])/len(y_train[:,1]),2),
                           'cancer_ratio_test':round(sum(y_test[:,1])/len(y_test[:,1]),2),
                           'accuracy': round(ev[1], 2),
                           'train_dataset_size':len(y_test),
                           'test_dataset_size':len(y_train),
                           'learning_rate':c['learning_rate'],
                           'batch_size':c['batch_size'],
                           'optimizer':c['optimizer'],
                           'elapsed': elapsed.seconds//3600}
    
                histories = histories.append(new_row, ignore_index=True)
                histories.to_csv('results.csv', mode='a', header=False, index=False)
                         
                logger.info(new_row)
            
        
    return 1


def main_loop(_epochs, _iters):
    logger.info("starting...")
    
    random.seed(123)
    np.random.seed(123)
    tf.keras.utils.set_random_seed(123)
    
    f = open('results.csv','w') 
    f.write("model_num, iter_num, cancer_filter, filter,  cancer_ratio_train, cancer_ratio_test, accuracy, train_dataset_size, test_dataset_size, learning_rate, batch_size, optimizer, elapsed\n")
    f.close()
    
    hist = train_model_multi_cv(_epochs, _iters, 'none', ['PTC'])
    
    logger.info("training finished!")
    
    
# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(utils.image_manipulator)

main_loop(20,3)


