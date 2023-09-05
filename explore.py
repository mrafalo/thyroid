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
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from scipy.stats import chi2_contingency
from sklearn import tree

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go

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
            random_state = 100,max_depth=6, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini
      
# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 6, min_samples_leaf = 5)
  
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

 
def chi2(_data):
    stat, p, dof, expected = chi2_contingency(_data)
    return p
    
def dec_tree(_data, _zmienne):
    
    X = _data.loc[:, _zmienne]
    Y = _data.loc[:, ['rak']]
        
    df1 = _data[_data.label_cancer.isin(['PTC'])]
    df2 = _data[_data.rak == 0]
    df = pd.concat([df1,df2])
    
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)
    
    
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
      
    m1_pred = clf_gini.predict(X_test)


    m1_pred_proba = clf_gini.predict_proba(X_test)[:,1]
    roc_plot(y_test, m1_pred_proba)


    print("Accuracy:",metrics.accuracy_score(y_test, m1_pred))

    return clf_gini

    # # Operational Phase
    # print("Results Using Gini Index:")
      
    # # Prediction using gini
    # y_pred_gini = prediction(X_test, clf_gini);
    # cal_accuracy(y_test, y_pred_gini)
      
    # print("Results Using Entropy:")
    # # Prediction using entropy
    # y_pred_entropy = prediction(X_test, clf_entropy)
    # cal_accuracy(y_test, y_pred_entropy)
    

def roc_plot(y_test, y_pred):
  
  m1_fpr, m1_tpr, _ = metrics.roc_curve(y_test,  y_pred)
  m1_auc = metrics.roc_auc_score(y_test, y_pred)

  plt.plot(m1_fpr,m1_tpr,label="model 1, auc="+str(round(m1_auc,2)))
  plt.legend(loc=4)
  plt.show()
  
  
df = d.load_data_file(BASE_FILE_PATH)

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

df[zmienne] = df[zmienne].astype(int)


for z in zmienne:
    contigency= pd.crosstab(df['rak'], df[z])
    p_val = chi2(contigency)
    if p_val < 0.05:
        print(z)
        print(contigency)


m1 = dec_tree(df, zmienne)

X = df[zmienne] # zmienne objaśniające
Y = df.rak # zmienna objaśniana
aucs = []
iters = []
_iters = 100
for i in range(0, _iters):
  
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
  m1 =DecisionTreeClassifier(criterion = "gini", max_depth=6, min_samples_leaf=5)
  m1.fit(x_train, y_train)

  m1_pred = m1.predict(x_test)
  m1_pred_proba = m1.predict_proba(x_test)[:,1]
  m1_auc = roc_auc_score(y_test, m1_pred_proba)
  aucs.append(m1_auc)
  iters.append(i)

  
