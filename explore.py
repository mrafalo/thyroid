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

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



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

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=8, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini
      
# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 8, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
  
  
# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
      
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    #print("Confusion Matrix: ",
    #confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))
  
    
df = d.load_data_file(BASE_FILE_PATH)
len(df)        


zmienne = ['echo_nieznacznie hipo', 'echo_gleboko hipo', 'echo_hiperechogeniczna',
'echo_izoechogeniczna', 'echo_mieszana', 'budowa_lita',
'budowa_lito_plynowa', 'budowa_plynowo_lita', 'ksztalt_owalny',
'ksztalt_okragly', 'ksztalt_nieregularny', 'orientacja_rownolegla',
'granice_rowne', 'granice_zatarte', 'granice_nierowne', 'brzegi_katowe',
'brzegi_mikrolobularne', 'brzegi_spikularne', 'halo', 'halo_cienka',
'halo_gruba ', 'Zwapnienia_mikrozwapnienia',
'Zwapnienia_makrozwapnienia', 'Zwapnienia_makro_obrączkowate',
'Zwapnienia_artefakty_typu_ogona_komety', 'torbka_modelowanie',
'torebka_naciek', 'unaczynienie_brak', 'unaczynienie_obwodowe',
'unaczynienie_centralne', 'unaczynienie_mieszane', 
'wezly_chlonne_patologiczne']


df = df.fillna(0)
X = df.loc[:, zmienne]
Y = df.loc[:, ['rak']]

df1 = df[df.label_cancer.isin(['PTC'])]
df2 = df[df.rak == 0]
df = pd.concat([df1,df2])

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


clf_gini = train_using_gini(X_train, X_test, y_train)
clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
  
# Operational Phase
print("Results Using Gini Index:")
  
# Prediction using gini
y_pred_gini = prediction(X_test, clf_gini);
cal_accuracy(y_test, y_pred_gini)
  
print("Results Using Entropy:")
# Prediction using entropy
y_pred_entropy = prediction(X_test, clf_entropy)
cal_accuracy(y_test, y_pred_entropy)
