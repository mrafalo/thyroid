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

df = pd.read_csv(BASE_FILE_PATH, sep=';')

def label_cancer (row):
   if row['HP_PTC'] == 1 :
      return 'PTC'
   if row['HP_FTC'] == 1 :
      return 'FTC'
   if row['HP_Hurthlea'] == 1:
      return 'HURTHLEA'
   if row['HP_MTC']  == 1:
      return 'MTC'
   if row['HP_DOBRZE_ZROZNICOWANE'] == 1:
      return 'DOBRZE_ZROZNICOWANY'
   if row['HP_ANA'] == 1:
      return 'ANAPLASTYCZNY'
   if row['HP_PLASKO'] == 1:
      return 'PLASKONABLONKOWY'
   else:
    return 'BENIGN'

df.columns
df['label_cancer'] = df.apply (lambda row: label_cancer(row), axis=1)


df.groupby(['label_cancer']).size().groupby(level=0).max() 


df.groupby(['HP_PTC']).size().groupby(level=0).max() 




row = [{'A':'X11', 'B':'X112', 'C':'X113'}]
df = pd.DataFrame(row)
df.to_csv('my_file.csv', mode='a', header=False, index=False)

