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
#from keras import backend as K
#import keras

from tensorflow.keras.optimizers import RMSprop, Adam
import yaml    
import logging
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix
#from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn import metrics
from scipy.stats import chi2_contingency
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn import tree
from tabulate import tabulate

import plotly

import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import kaleido
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


BATCH_SIZE = 12
BASE_LR = 0.001#1e-6
SEED = 123


ZMIENNE = ['echo_nieznacznie_hipo', 'echo_gleboko_hipo', 'echo_hiperechogeniczna',
       'echo_izoechogeniczna', 'echo_mieszana', 'budowa_lita',
       'budowa_lito_plynowa', 'budowa_plynowo_lita', 'ksztalt_owalny',
       'ksztalt_okragly', 'ksztalt_nieregularny', 'orientacja_rownolegla',
       'granice_rowne', 'granice_zatarte', 'granice_nierowne', 'brzegi_katowe',
       'brzegi_mikrolobularne', 'brzegi_spikularne', 'halo', 'halo_cienka',
       'halo_gruba', 'Zwapnienia_mikrozwapnienia',
       'Zwapnienia_makrozwapnienia', 'Zwapnienia_makro_obrączkowate',
       'Zwapnienia_artefakty_typu_ogona_komety', 'torbka_modelowanie',
       'torebka_naciek', 'unaczynienie_brak', 'unaczynienie_obwodowe',
       'unaczynienie_centralne', 'unaczynienie_mieszane', 'USG_AZT',
       'wezly_chlonne_patologiczne']
    
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

def random_forest_cv_PTC(_iters):
    
    df = d.load_data_file(BASE_FILE_PATH)
    df1 = df[df.rak == 1]
    df2 = df[df.rak == 0]
    df = pd.concat([df1,df2])
    
    X = df[ZMIENNE] # zmienne objaśniające
    y = df.rak # zmienna objaśniana
    aucs = []
    
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
      
      aucs.append(m1_auc)
      # m2 = XGBClassifier(use_label_encoder=False)
      # m2 = m1.fit(x_train,y_train)
      # m2_pred_proba = m2.predict_proba(x_test)[:,1]
      # m2_auc = roc_auc_score(y_test, m2_pred_proba)
      # m2_fpr, m2_tpr, t = metrics.roc_curve(y_test,  m2_pred_proba)
      # m2_optimal_idx = np.argmax(m2_tpr - m2_fpr)
      # m2_optimal_threshold = t[m2_optimal_idx]
      # m2_pred_proba[m2_pred_proba < m2_optimal_threshold] = 0
      # m2_pred_proba[m2_pred_proba >= m2_optimal_threshold] = 1
      # m2_recall = accuracy_score(y_test, m2_pred_proba)
      
      print('forest: recall =', round(m1_recall,2), " auc =", round(m1_auc,2))
      
      
    return aucs

def report_variables_vs_typ_raka():
    df = d.load_data_file(BASE_FILE_PATH)
    
    for z in ZMIENNE:
        tmp = df.loc[df[z]>=0,]
        contigency= pd.crosstab(tmp['label_cancer'], tmp[z])
        p_val = chi2(contigency)


def report_variables_vs_nieokreslone():
    df = d.load_data_file(BASE_FILE_PATH)
    
    for z in ZMIENNE:
        tmp = df.loc[df[z]>=0,]
        contigency= pd.crosstab(tmp['HP_NIEOKRESLONE'], tmp[z])
        p_val = chi2(contigency)


def report_variables_vs_PTC():
    df = d.load_data_file(BASE_FILE_PATH)
    df1 = df[df.label_cancer.isin(['PTC'])]
    df2 = df[df.rak == 0]
    df = pd.concat([df1,df2])
    
    print("Liczba pacjentów: ", len(df), "liczba PTC:", len(df[df.rak == 1]))
        
    for z in ZMIENNE:
        tmp = df.loc[df[z]>=0,]
        contigency = pd.crosstab(tmp['HP_PTC'], tmp[z])
        p_val = chi2(z, contigency)

        if p_val < 0.05:
            #print(z, round(p_val,3))
            x11=contigency[0][0]
            x12=contigency[1][0]
            x21=contigency[0][1]
            x22=contigency[1][1]
        
            #print(contigency)
            print("Wsród pacjentów z ",z, " ", round(x22/(x12+x22)*100), "% ma raka PTC", sep="")
            print("Wsród pacjentów z ",z, " ", round(x12/(x12+x22)*100), "% nie ma raka PTC", sep="")
            print("Wsród pacjentów z rakiem PTC ",round(x22/(x21+x22)*100), "% ma ", z, sep="")
            print("Wsród pacjentów bez raka ",round(x12/(x12+x11)*100), "% ma ", z, sep="")
            print("--------------------")




# def xgb_model():             
    
#     df = d.load_data_file(BASE_FILE_PATH)
#     df1 = df[df.label_cancer.isin(['PTC'])]
#     df2 = df[df.rak == 0]
#     df = pd.concat([df1,df2])

#     X = df[ZMIENNE] # zmienne objaśniające
#     y = df.rak # zmienna objaśniana
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
#     np.random.seed(123)
#     m2 = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', )
#     m2 = m2.fit(x_train,y_train)
#     m2_pred_proba = m2.predict_proba(x_test)[:,1]
#     m2_auc = roc_auc_score(y_test, m2_pred_proba)
#     m2_fpr, m2_tpr, t = metrics.roc_curve(y_test,  m2_pred_proba)
#     m2_optimal_idx = np.argmax(m2_tpr - m2_fpr)
#     m2_optimal_threshold = t[m2_optimal_idx]
#     m2_pred_proba[m2_pred_proba < m2_optimal_threshold] = 0
#     m2_pred_proba[m2_pred_proba >= m2_optimal_threshold] = 1
#     m2_recall = accuracy_score(y_test, m2_pred_proba)
    
#     dataset = pd.DataFrame({'feature': m2.get_booster().feature_names, 'importance': np.round(m2.feature_importances_,2)})
#     dataset=dataset.sort_values('importance', ascending=False).head(10)
#     print("AUC:", m2_auc)
#     print(dataset)
      
def forest_model():
    df = d.load_data_file(BASE_FILE_PATH)
    df1 = df[df.label_cancer.isin(['PTC'])]
    df2 = df[df.rak == 0]
    df = pd.concat([df1,df2])

    X = df[ZMIENNE] # zmienne objaśniające
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
    for feature, importance in zip(ZMIENNE, m2.feature_importances_):
        feats[feature] = np.round(importance,2) 
    
    dataset = pd.DataFrame(feats.items(), columns=['feature', 'importance'])
    dataset=dataset.sort_values('importance', ascending=False).head(10)
    print("AUC:", m2_auc)
    print("TPR:", m2_tpr)
    print(dataset)
          



def report_overview(_latex = False):
    df = d.load_data_file(BASE_FILE_PATH)    
    print("Liczba pacjentów: ", len(df), "liczba nowotworow złoliwych:", len(df[df.rak == 1]), "liczba łagodnych:", len(df[df.label_cancer == "BENIGN"]))    
    
    
    print("Pacjenci wg rodzaju nowotworu i Bethesda:")
    tmp = pd.crosstab(df.rak, df.BACC_Bethesda, margins = False) 
    if _latex:
        print(tmp.to_latex())
    else:
        print(tabulate(tmp,headers='firstrow',tablefmt='html'))
        html = tabulate(tmp,headers='firstrow',tablefmt='html')
        text_file = open("out.html", "w") 
        text_file.write(html) 
        text_file.close() 


    print("Pacjenci wg rodzaju nowotworu i tirads:")
    tmp = pd.crosstab(df.rak, df.tirads, margins = False) 
    if _latex:
        print(tmp.to_latex())
    else:
        print(tabulate(tmp,headers='firstrow',tablefmt='html'))
        html = tabulate(tmp,headers='firstrow',tablefmt='html')
        text_file = open("out.html", "w") 
        text_file.write(html) 
        text_file.close() 
        
        

    print("Pacjenci wg płci:")
    tmp = pd.crosstab(df.rak, df.plec, margins = False) 
    if _latex:
        print(tmp.to_latex())
    else:
        print(tabulate(tmp,headers='firstrow',tablefmt='grid'))
    
    print("Pacjenci wg cech:")
    
  
    vars = []
    cancer1 = []
    cancer0 = []
    for z in ZMIENNE:
        tmp = df.groupby(z).size().reset_index(name='cnt')
        vars.append(z)
        cancer1.append(tmp.at[1,"cnt"])
        cancer0.append(tmp.at[0,"cnt"])

    tmp = pd.DataFrame({
        'Feature': vars,
        'Benign': cancer0,
        'Malignant': cancer1
        })
    
    if _latex:
        print(tmp.to_latex(),'\n')
    else:
        print(z,'\n')
        print(tmp,'\n')
        
    return df


df = report_overview(True)

tmp = df.groupby("Zwapnienia_mikrozwapnienia").size().reset_index(name='cnt')
tmp.at[1,"cnt"]


tmp = df.groupby(['tirads']).agg({'max_dim': ['mean', 'count']})
tmp.columns = [ ' '.join(str(i) for i in col) for col in tmp.columns]
tmp.reset_index(inplace=True)
fig = px.line(tmp, x='Customer_Age', y='Months_on_book mean', color="Card_Category", title="sample figure")
fig.show()

# fillcolor='rgba(26,150,65,0.5)'
fig = px.histogram(df, x="max_dim", color="rak", 
                   color_discrete_map = {0:'rgba(26,150,65,0.5)',1:'rgba(150,25,65,0.5)'},
                   barmode='overlay')



# rozkład wieku (histogram)

# rozklady wymiarow - gestosci

fig = px.box(df, x="rak", y="max_dim")
fig.update_xaxes(title_text='Malignancy')
fig.update_yaxes(title_text='Tumor size [mm]')
fig.show()


auc = random_forest_cv_PTC(100);
np.mean(auc)
