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
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn import tree

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


def chi2(_data):
    stat, p, dof, expected = chi2_contingency(_data)
    
    if p_val < 0.05:
        print(z, round(p_val,3))
        print(contigency)
        print("--------------------")
    return p
      

def roc_plot(y_test, y_pred):
  
  m1_fpr, m1_tpr, _ = metrics.roc_curve(y_test,  y_pred)
  m1_auc = metrics.roc_auc_score(y_test, y_pred)

  plt.plot(m1_fpr,m1_tpr,label="model 1, auc="+str(round(m1_auc,2)))
  plt.legend(loc=4)
  plt.show()
  

def decision_tree_cv_ptc(_data, _iters, _zmienne):
    
    df1 = _data[_data.label_cancer.isin(['PTC'])]
    df2 = _data[_data.rak == 0]
    df = pd.concat([df1,df2])
    
    X = df[_zmienne] # zmienne objaśniające
    y = df.rak # zmienna objaśniana
    
    np.random.seed(123)
    
    for i in range(0, _iters):
       x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
       
       m1 = DecisionTreeClassifier(criterion = "gini",max_depth=4, min_samples_leaf=5)
       m1 = m1.fit(x_train,y_train)
       m1_pred_proba = m1.predict_proba(x_test)[:,1]
       m1_auc = roc_auc_score(y_test, m1_pred_proba)
       m1_fpr, m1_tpr, t = metrics.roc_curve(y_test,  m1_pred_proba)
       m1_optimal_idx = np.argmax(m1_tpr - m1_fpr)
       m1_optimal_threshold = t[m1_optimal_idx]
       m1_pred_proba[m1_pred_proba < m1_optimal_threshold] = 0
       m1_pred_proba[m1_pred_proba >= m1_optimal_threshold] = 1
       m1_recall = accuracy_score(y_test, m1_pred_proba)
       
       print('forest: recall =', round(m1_recall,2), " auc =", round(m1_auc,2))
      
    return m1

def random_forest_cv_ptc(_data, _iters, _zmienne):
    
    df1 = _data[_data.label_cancer.isin(['PTC'])]
    df2 = _data[_data.rak == 0]
    df = pd.concat([df1,df2])
    
    X = df[_zmienne] # zmienne objaśniające
    y = df.rak # zmienna objaśniana
    
    np.random.seed(123)

    for i in range(0, _iters):
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
      
      m1 = RandomForestClassifier()
      m1 = m1.fit(x_train,y_train)
      m1_pred_proba = m1.predict_proba(x_test)[:,1]
      m1_auc = roc_auc_score(y_test, m1_pred_proba)
      m1_fpr, m1_tpr, t = metrics.roc_curve(y_test,  m1_pred_proba)
      m1_optimal_idx = np.argmax(m1_tpr - m1_fpr)
      m1_optimal_threshold = t[m1_optimal_idx]
      m1_pred_proba[m1_pred_proba < m1_optimal_threshold] = 0
      m1_pred_proba[m1_pred_proba >= m1_optimal_threshold] = 1
      m1_recall = accuracy_score(y_test, m1_pred_proba)
      
      
      m2 = XGBClassifier()
      m2 = m1.fit(x_train,y_train)
      m2_pred_proba = m2.predict_proba(x_test)[:,1]
      m2_auc = roc_auc_score(y_test, m2_pred_proba)
      m2_fpr, m2_tpr, t = metrics.roc_curve(y_test,  m2_pred_proba)
      m2_optimal_idx = np.argmax(m2_tpr - m2_fpr)
      m2_optimal_threshold = t[m2_optimal_idx]
      m2_pred_proba[m2_pred_proba < m2_optimal_threshold] = 0
      m2_pred_proba[m2_pred_proba >= m2_optimal_threshold] = 1
      m2_recall = accuracy_score(y_test, m2_pred_proba)
      
      print('forest: recall =', round(m1_recall,2), " auc =", round(m1_auc,2), 'XGB: recall =', round(m2_recall,2), " auc =", round(m2_auc,2))


def random_forest_cv_rak(_data, _iters, _zmienne):
    
    X = _data[_zmienne] # zmienne objaśniające
    y = _data.rak # zmienna objaśniana
    
    np.random.seed(123)

    for i in range(0, _iters):
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
      
      m1 = RandomForestClassifier()
      m1 = m1.fit(x_train,y_train)
      m1_pred_proba = m1.predict_proba(x_test)[:,1]
      m1_auc = roc_auc_score(y_test, m1_pred_proba)
      m1_fpr, m1_tpr, t = metrics.roc_curve(y_test,  m1_pred_proba)
      m1_optimal_idx = np.argmax(m1_tpr - m1_fpr)
      m1_optimal_threshold = t[m1_optimal_idx]
      m1_pred_proba[m1_pred_proba < m1_optimal_threshold] = 0
      m1_pred_proba[m1_pred_proba >= m1_optimal_threshold] = 1
      m1_recall = accuracy_score(y_test, m1_pred_proba)
      
      
      m2 = XGBClassifier()
      m2 = m1.fit(x_train,y_train)
      m2_pred_proba = m2.predict_proba(x_test)[:,1]
      m2_auc = roc_auc_score(y_test, m2_pred_proba)
      m2_fpr, m2_tpr, t = metrics.roc_curve(y_test,  m2_pred_proba)
      m2_optimal_idx = np.argmax(m2_tpr - m2_fpr)
      m2_optimal_threshold = t[m2_optimal_idx]
      m2_pred_proba[m2_pred_proba < m2_optimal_threshold] = 0
      m2_pred_proba[m2_pred_proba >= m2_optimal_threshold] = 1
      m2_recall = accuracy_score(y_test, m2_pred_proba)
      
      print('forest: recall =', round(m1_recall,2), " auc =", round(m1_auc,2), 'XGB: recall =', round(m2_recall,2), " auc =", round(m2_auc,2))


def report_variables_vs_typ_raka():
    df = d.load_data_file(BASE_FILE_PATH)

    zmienne = ['echo_nieznacznie_hipo', 'echo_gleboko_hipo', 'echo_hiperechogeniczna',
    'echo_izoechogeniczna', 'echo_mieszana', 'budowa_lita',
    'budowa_lito_plynowa', 'budowa_plynowo_lita', 'ksztalt_owalny',
    'ksztalt_okragly', 'ksztalt_nieregularny', 'orientacja_rownolegla',
    'granice_rowne', 'granice_zatarte', 'granice_nierowne', 'brzegi_katowe',
    'brzegi_mikrolobularne', 'brzegi_spikularne', 'halo', 'halo_cienka',
    'halo_gruba ', 'Zwapnienia_mikrozwapnienia',
    'Zwapnienia_makrozwapnienia', 'Zwapnienia_makro_obrączkowate',
    'Zwapnienia_artefakty_typu_ogona_komety', 'torbka_modelowanie',
    'torebka_naciek', 'unaczynienie_brak', 'unaczynienie_obwodowe',
    'unaczynienie_centralne', 'unaczynienie_mieszane', 'USG_AZT',
    'wezly_chlonne_patologiczne',
    'lokalizacja_prawy_plat', 'lokalizacja_lewy_plat', 'lokalizacja_ciesn']
    
    for z in zmienne:
       
        tmp = df.loc[df[z]>=0,]
        contigency= pd.crosstab(tmp['label_cancer'], tmp[z])
        p_val = chi2(contigency)


def report_variables_vs_nieokreslone():
    df = d.load_data_file(BASE_FILE_PATH)

    zmienne = ['echo_nieznacznie_hipo', 'echo_gleboko_hipo', 'echo_hiperechogeniczna',
    'echo_izoechogeniczna', 'echo_mieszana', 'budowa_lita',
    'budowa_lito_plynowa', 'budowa_plynowo_lita', 'ksztalt_owalny',
    'ksztalt_okragly', 'ksztalt_nieregularny', 'orientacja_rownolegla',
    'granice_rowne', 'granice_zatarte', 'granice_nierowne', 'brzegi_katowe',
    'brzegi_mikrolobularne', 'brzegi_spikularne', 'halo', 'halo_cienka',
    'halo_gruba ', 'Zwapnienia_mikrozwapnienia',
    'Zwapnienia_makrozwapnienia', 'Zwapnienia_makro_obrączkowate',
    'Zwapnienia_artefakty_typu_ogona_komety', 'torbka_modelowanie',
    'torebka_naciek', 'unaczynienie_brak', 'unaczynienie_obwodowe',
    'unaczynienie_centralne', 'unaczynienie_mieszane', 'USG_AZT',
    'wezly_chlonne_patologiczne',
    'lokalizacja_prawy_plat', 'lokalizacja_lewy_plat', 'lokalizacja_ciesn']
    
    for z in zmienne:
       
        tmp = df.loc[df[z]>=0,]
        contigency= pd.crosstab(tmp['HP_NIEOKRESLONE'], tmp[z])
        p_val = chi2(contigency)


def report_variables_vs_PTC():
    df = d.load_data_file(BASE_FILE_PATH)

    zmienne = ['echo_nieznacznie_hipo', 'echo_gleboko_hipo', 'echo_hiperechogeniczna',
    'echo_izoechogeniczna', 'echo_mieszana', 'budowa_lita',
    'budowa_lito_plynowa', 'budowa_plynowo_lita', 'ksztalt_owalny',
    'ksztalt_okragly', 'ksztalt_nieregularny', 'orientacja_rownolegla',
    'granice_rowne', 'granice_zatarte', 'granice_nierowne', 'brzegi_katowe',
    'brzegi_mikrolobularne', 'brzegi_spikularne', 'halo', 'halo_cienka',
    'halo_gruba ', 'Zwapnienia_mikrozwapnienia',
    'Zwapnienia_makrozwapnienia', 'Zwapnienia_makro_obrączkowate',
    'Zwapnienia_artefakty_typu_ogona_komety', 'torbka_modelowanie',
    'torebka_naciek', 'unaczynienie_brak', 'unaczynienie_obwodowe',
    'unaczynienie_centralne', 'unaczynienie_mieszane', 'USG_AZT',
    'wezly_chlonne_patologiczne',
    'lokalizacja_prawy_plat', 'lokalizacja_lewy_plat', 'lokalizacja_ciesn']
    
    for z in zmienne:
       
        tmp = df.loc[df[z]>=0,]
        contigency= pd.crosstab(tmp['HP_PTC'], tmp[z])
        p_val = chi2(contigency)

            
report_variables_vs_nieokreslone()

# m1 = decision_tree_cv_ptc(df, 10, zmienne)
# print('---')
# random_forest_cv_ptc(df, 10, zmienne)
# print('---')
# random_forest_cv_rak(df, 10, zmienne)
# plt.figure(figsize=(12,12)) 
# tree.plot_tree(m1,
#            feature_names = zmienne, 
#            class_names=['0', '1'],
#            fontsize=12,
#            filled = True);

