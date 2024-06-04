import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import work.models as m
import importlib
import work
import work.models as m
import work.data as d
import utils
import utils.image_manipulator as im
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
import yaml    
import logging
from matplotlib import pyplot as plt
import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from scipy.stats import chi2_contingency
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import keras.backend as K

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn import tree
from tabulate import tabulate

import plotly

import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

pio.renderers.default='svg'


with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    BASE_PATH = cfg['BASE_PATH']
    BASE_FILE_PATH = cfg['BASE_FILE_PATH']
    MODELING_PATH = cfg['MODELING_PATH']
    RAW_INPUT_PATH = cfg['RAW_INPUT_PATH']
    ANNOTATION_INPUT_PATH = cfg['ANNOTATION_INPUT_PATH']
    MODELING_INPUT_PATH = cfg['MODELING_INPUT_PATH']
    
    IMG_PATH_BASE = cfg['IMG_PATH_BASE']
    IMG_PATH_BW = cfg['IMG_PATH_BW']
    IMG_PATH_SOBEL = cfg['IMG_PATH_SOBEL']
    IMG_PATH_HEAT = cfg['IMG_PATH_HEAT']
    IMG_PATH_CANNY = cfg['IMG_PATH_CANNY']
    IMG_PATH_FELZEN = cfg['IMG_PATH_FELZEN']
    
    IMG_WIDTH = cfg['IMG_WIDTH']
    IMG_HEIGHT = cfg['IMG_HEIGHT']
    CV_ITERATIONS = cfg['CV_ITERATIONS']
    EPOCHS = cfg['EPOCHS']
    SEED = cfg['SEED']


RESULT_FILE = 'results/results6.csv'

def model_summary(_model_name):
    df = pd.read_csv(RESULT_FILE, sep=';')
    res = df.loc[(df['model_name'] == _model_name) & 
                 (df['optimizer']=='SGD') &
                 (df['learning_rate']==0.005),:].groupby('loss_function').agg({
        'auc_max': 'max',
        'sensitivity_max': 'max',
        'precision_max': 'max'
        })
    
    res = res.round(2)
    print(res.to_latex())
    return res
    


def get_number_of_params(_models, _latex=True):
    df = pd.read_csv(RESULT_FILE, sep=';')   

    res = df.loc[df['model_name'].isin(_models),:].groupby('model_name').agg({'iterations': 'sum'})
                         
    params = []
    layers = []
    for model_name in _models:
        m1 = m.get_model_by_name(model_name, IMG_WIDTH, IMG_HEIGHT)
        number_of_params = sum([np.prod(K.get_value(w).shape) for w in m1.trainable_weights])
        params.append(number_of_params)
        layers.append(len(m1.layers))
        
    res['number_of_parameters'] = params
    res['number_of_layers'] = layers
     
    if _latex:
        print(res.to_latex())
    else:
        print(res)
        
        
        
def models_summary(_models, _latex=True):
    df = pd.read_csv(RESULT_FILE, sep=';')

    losses = ['focal_loss', 'kl_divergence', 'categorical_crossentropy', 'squared_hinge' ]
    
    res = df.loc[df['loss_function'].isin(losses),:].pivot_table(
        index='model_name', 
        columns='loss_function', 
        values='auc_max', 
        aggfunc=np.max
    )

    
    res = res.round(2)
    
    if _latex:
        print(res.to_latex())
    else:
        print(res.to_string())

def get_optimal_models(_latex=True):
    df = pd.read_csv(RESULT_FILE, sep=';')
    cols = ['model_name', 'optimizer', 'loss_function', 'auc_max', 'sensitivity_max', 'precision_max']
    df = df[cols]
    idx_auc_max = df.groupby('model_name')['auc_max'].idxmax()

    res = df.loc[idx_auc_max]
    
    res = res.round(2)

    if _latex:
        print(res.to_latex(index=False))
    else:
        print(res.to_string(index=False))

models = ['ResNet101', 'ResNet152', 'ResNet50', 'VGG16', 'VGG19', 'cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5']

get_optimal_models()

get_number_of_params(models, True)

models_summary(models, True)




m1 = m.get_model_by_name('cnn1',IMG_WIDTH, IMG_HEIGHT)
keras.utils.plot_model(m1, to_file='figures/cnn1.png', show_shapes=True)


# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(utils.image_manipulator)
    