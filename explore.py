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


def chi2(_var, _data):
    stat, p_val, dof, expected = chi2_contingency(_data)
    
    return p_val
      

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

def random_forest_cv_ptc(_data, _iters):
    
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
                 
    
    df1 = _data[_data.label_cancer.isin(['PTC'])]
    df2 = _data[_data.rak == 0]
    df = pd.concat([df1,df2])

    X = df[zmienne] # zmienne objaśniające
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
      m2 = m2.fit(x_train,y_train)
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
      
      
      m2 = XGBClassifier(use_label_encoder=False)
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
    df1 = df[df.label_cancer.isin(['PTC'])]
    df2 = df[df.rak == 0]
    df = pd.concat([df1,df2])
    
    print("Liczba pacjentów: ", len(df), "liczba PTC:", len(df[df.rak == 1]))
    
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
    'wezly_chlonne_patologiczne']
    
    for z in zmienne:
        tmp = df.loc[df[z]>=0,]
        contigency = pd.crosstab(tmp['HP_PTC'], tmp[z])
        p_val = chi2(z, contigency)


        if p_val < 0.05:
            print(z, round(p_val,3))
            x11=contigency[0][0]
            x12=contigency[1][0]
            x21=contigency[0][1]
            x22=contigency[1][1]
        
            print(contigency)
            print("Wsród pacjentów z ",z, " ", round(x22/(x12+x22)*100), "%, ma raka PTC", sep="")
            print("Wsród pacjentów z ",z, " ", round(x12/(x12+x22)*100), "%, nie ma raka PTC", sep="")
            print("Wsród pacjentów z rakiem PTC ",round(x22/(x21+x22)*100), "%, ma ", z, sep="")
            print("Wsród pacjentów bez raka ",round(x12/(x12+x11)*100), "%, ma ", z, sep="")
            print("Wsród pacjentów z rakiem PTC ",round(x22/(x21+x22)*100), "%, ma ", z, sep="")
            print("--------------------")


def xgb_model(_data):
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
           'wezly_chlonne_patologiczne']
             
    
    df1 = _data[_data.label_cancer.isin(['PTC'])]
    df2 = _data[_data.rak == 0]
    df = pd.concat([df1,df2])

    X = df[zmienne] # zmienne objaśniające
    y = df.rak # zmienna objaśniana
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    np.random.seed(123)
    m2 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', )
    m2 = m2.fit(x_train,y_train)
    m2_pred_proba = m2.predict_proba(x_test)[:,1]
    m2_auc = roc_auc_score(y_test, m2_pred_proba)
    m2_fpr, m2_tpr, t = metrics.roc_curve(y_test,  m2_pred_proba)
    m2_optimal_idx = np.argmax(m2_tpr - m2_fpr)
    m2_optimal_threshold = t[m2_optimal_idx]
    m2_pred_proba[m2_pred_proba < m2_optimal_threshold] = 0
    m2_pred_proba[m2_pred_proba >= m2_optimal_threshold] = 1
    m2_recall = accuracy_score(y_test, m2_pred_proba)
    
    dataset = pd.DataFrame({'feature': m2.get_booster().feature_names, 'importance': np.round(m2.feature_importances_,2)})
    dataset=dataset.sort_values('importance', ascending=False).head(10)
    print(dataset)
    # xgb_model = xgboost.XGBClassifier(num_class=7,
    #                               learning_rate=0.1,
    #                               num_iterations=1000,
    #                               max_depth=10,
    #                               feature_fraction=0.7, 
    #                               scale_pos_weight=1.5,
    #                               boosting='gbdt',
    #                               metric='multiclass',
    #                               eval_metric='mlogloss')
      
def forest_model(_data):
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
           'wezly_chlonne_patologiczne']
             
    df1 = _data[_data.label_cancer.isin(['PTC'])]
    df2 = _data[_data.rak == 0]
    df = pd.concat([df1,df2])

    X = df[zmienne] # zmienne objaśniające
    y = df.rak # zmienna objaśniana
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    np.random.seed(123)
    m2 = RandomForestClassifier()
    m2 = m2.fit(x_train,y_train)
    m2_pred_proba = m2.predict_proba(x_test)[:,1]
    m2_auc = roc_auc_score(y_test, m2_pred_proba)
    m2_fpr, m2_tpr, t = metrics.roc_curve(y_test,  m2_pred_proba)
    m2_optimal_idx = np.argmax(m2_tpr - m2_fpr)
    m2_optimal_threshold = t[m2_optimal_idx]
    m2_pred_proba[m2_pred_proba < m2_optimal_threshold] = 0
    m2_pred_proba[m2_pred_proba >= m2_optimal_threshold] = 1
    m2_recall = accuracy_score(y_test, m2_pred_proba)
   
    feats = {} 
    for feature, importance in zip(zmienne, m2.feature_importances_):
        feats[feature] = np.round(importance,2) 
    
    dataset = pd.DataFrame(feats.items(), columns=['feature', 'importance'])
    dataset=dataset.sort_values('importance', ascending=False).head(10)
    print(dataset)
          
report_variables_vs_PTC()


df = d.load_data_file(BASE_FILE_PATH)

random_forest_cv_ptc(df, 20)

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
       'wezly_chlonne_patologiczne']
         

df1 = df[df.label_cancer.isin(['PTC'])]
df2 = df[df.rak == 0]
df = pd.concat([df1,df2])

forest_model(df)
xgb_model(df)


X = df[zmienne] # zmienne objaśniające
y = df.rak # zmienna objaśniana
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

np.random.seed(123)
m2 = RandomForestClassifier()
m2 = m2.fit(x_train,y_train)
m2_pred_proba = m2.predict_proba(x_test)[:,1]
m2_auc = roc_auc_score(y_test, m2_pred_proba)
m2_fpr, m2_tpr, t = metrics.roc_curve(y_test,  m2_pred_proba)
m2_optimal_idx = np.argmax(m2_tpr - m2_fpr)
m2_optimal_threshold = t[m2_optimal_idx]
m2_pred_proba[m2_pred_proba < m2_optimal_threshold] = 0
m2_pred_proba[m2_pred_proba >= m2_optimal_threshold] = 1
m2_recall = accuracy_score(y_test, m2_pred_proba)


# granice_rowne
# Zwapnienia_mikrozwapnienia
# granice_zatarte
# echo_gleboko_hipo
# USG_AZT
# ksztalt_nieregularny
# Zwapnienia_makrozwapnienia
# granice_nierowne
# echo_nieznacznie_hipo
# unaczynienie_mieszane
# wezly_chlonne_patologiczne
# brzegi_spikularne
# ksztalt_owalny
# torbka_modelowanie
