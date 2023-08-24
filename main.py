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
            
            
def train_model_multi_cv(_epochs, _iters, _filter="none", _cancer_filter = 'none'):
    
    
    if _filter == "none": 
        INPUT_PATH = BASE_PATH + 'modeling/all_images/'
        OUTPUT_TEST_PATH = TEST_PATH
        OUTPUT_TRAIN_PATH = TRAIN_PATH
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
    
    accuracies = []
    cancer_ratios_train = []
    cancer_ratios_test = []
    model_nums = []
    iter_nums = []
    train_dataset_sizes = []
    test_dataset_sizes = []
    filters = []
    cancer_filters = []
    
    models = m.model_sequence_manual_2(IMG_WIDTH, IMG_HEIGHT)
    model_cnt = len(models)
    
    logger.info('processing start... ' + 'model: ' + str(model_cnt) + ' cv iters: ' + str(_iters) + ' filter: ' + str(_filter))
    
    for m_num in range(model_cnt):
        
        
        for i in range(_iters):
            
            accuracies = []
            cancer_ratios_train = []
            cancer_ratios_test = []
            model_nums = []
            iter_nums = []
            train_dataset_sizes = []
            test_dataset_sizes = []
            filters = []
            cancer_filters = []
    
            #logger.info('model: ' + str(m_num+1)  + "/" + str(model_cnt) +' iteration: ' + str(i+1) + "/" + str(_iters))
            models = m.model_sequence_manual_2(IMG_WIDTH, IMG_HEIGHT)
            m1 = models[m_num]
            
            keras.backend.clear_session()
                    
            d.split_files(INPUT_PATH, OUTPUT_TRAIN_PATH, OUTPUT_TEST_PATH, 0.15)
            X_train, y_train, X_test, y_test = d.split_data(BASE_FILE_PATH, OUTPUT_TRAIN_PATH, OUTPUT_TEST_PATH, 0, True, _cancer_filter)

            #logger.info('iteration: ' + str(i+1) + "/" + str(_iters))
            m1.compile(optimizer = Adam(learning_rate=BASE_LR), loss='categorical_crossentropy', metrics=["accuracy"]) 
            hist = m1.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=_epochs, verbose=False)
            
            
            ev = m1.evaluate(X_test, y_test, verbose=False)
             
            accuracies.append(round(ev[1], 2))
            cancer_ratios_train.append(round(sum(y_train[:,1])/len(y_train[:,1]),2))
            cancer_ratios_test.append(round(sum(y_test[:,1])/len(y_test[:,1]),2))
            
            train_dataset_sizes.append(len(y_test))
            test_dataset_sizes.append(len(y_train))
            filters.append(_filter)
            
            model_nums.append(m_num)
            iter_nums.append(i+1)
            cancer_filters.append(_cancer_filter)
            
            histories = pd.DataFrame(list(zip(cancer_filters, filters, model_nums, iter_nums, cancer_ratios_train, cancer_ratios_test,accuracies, train_dataset_sizes, test_dataset_sizes)), 
                                     columns =["cancer_filters", "filter", "model_nums", "iter_nums", "cancer_ratios_train", "cancer_ratios_test","accuracies", "train_dataset_sizes", "test_dataset_sizes"])
            
            histories.to_csv('results.csv', mode='a', header=False, index=False)
            
           
            logger.info('model: ' + str(m_num+1)  + "/" + str(model_cnt) +' iter: ' + str(i+1) + "/" + str(_iters) + " accuracy = "  + str(round(ev[1], 2)))
        
    
    return 1


def main_loop(_epochs, _iters):
    logger.info("starting...")
    tf.keras.utils.set_random_seed(123)
    random.seed(123)
    np.random.seed(123)
    
    f = open('results.csv','w') 
    f.write("filter, model_nums, iter_nums, cancer_ratios_train, cancer_ratios_test, accuracies, train_dataset_sizes,test_dataset_sizes\n")
    f.close()
    
    hist = train_model_multi_cv(_epochs, _iters, 'none', ['PTC'])
    hist = train_model_multi_cv(_epochs, _iters, 'bw', ['PTC'])
    hist = train_model_multi_cv(_epochs, _iters, 'heat', ['PTC'])
    hist = train_model_multi_cv(_epochs, _iters, 'canny', ['PTC'])
    hist = train_model_multi_cv(_epochs, _iters, 'felzen', ['PTC'])
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', ['PTC'])
    
    hist = train_model_multi_cv(_epochs, _iters, 'none', 'none')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'none')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'none')
    hist = train_model_multi_cv(_epochs, _iters, 'canny', 'none')
    hist = train_model_multi_cv(_epochs, _iters, 'felzen', 'none')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'none')
    
    logger.info("training finished!")
    
    
# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(utils.image_manipulator)

main_loop(20,5)

