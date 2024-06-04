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

from sklearn.metrics import confusion_matrix
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
    CV_ITERATIONS = cfg['CV_ITERATIONS']
    EPOCHS = cfg['EPOCHS']
    SEED = cfg['SEED']


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

ZMIENNE_BASE = ['echo_gleboko_hipo', 'echo_izoechogeniczna','budowa_lita', 'budowa_lito_plynowa', 'ksztalt_owalny', 
'ksztalt_nieregularny', 'granice_zatarte', 'brzegi_mikrolobularne', 'halo','Zwapnienia_mikrozwapnienia', 'Zwapnienia_makrozwapnienia', 'torebka_naciek', 
'unaczynienie_brak', 'unaczynienie_obwodowe', 'unaczynienie_mieszane']

ZMIENNE_ENG = ['hypoechoic', 'isoechoic','solid', 'fluid-filled', 'oval', 
'irregular', 'boundaries blurred', 'microlobular', 'halo','microcalcifications', 'macrocalcifications', 'infiltrative capsule', 
'no vascularity', 'peripheral vascularity', 'mixed vascularity']

def chi2(_data):
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

def random_forest_cv(_iters):
    
    df = d.load_data_file(BASE_FILE_PATH)
    df1 = df[df.rak == 1]
    df2 = df[df.rak == 0]
    df = pd.concat([df1,df2])
    
    X = df[ZMIENNE_BASE] # zmienne objaśniające
    y = df.rak # zmienna objaśniana
    r = []
    
    np.random.seed(123)

    for i in range(0, _iters):
      x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
      
      m1 = RandomForestClassifier()
      m1 = m1.fit(x_train,y_train)
      
      res = m.model_predictor_scikit(m1, x_test, y_test)
      
      r.append(res)

      print(res)
      
      
    return pd.DataFrame(r)

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
    tmp
    print(tmp.to_string(index=True),'\n')
    
    if _latex:
        print(tmp.to_latex())
    else:
        print(tabulate(tmp,headers='firstrow',tablefmt='grid'))
    
    print("Pacjenci wg cech:")
    
  
    vars = []
    cancer1 = []
    cancer0 = []
    p_values = []
    i = 0

   
    
    for z in ZMIENNE_BASE:
        tmp = df.loc[df[z]==1,].groupby('rak').size().reset_index(name='cnt')
        ct = pd.crosstab(df['rak'], df[z])
         
        if len(tmp) > 1:
            chi2_p_value = round(chi2(ct),3)
            #print(z, chi2_p_value)
            vars.append(ZMIENNE_ENG[i])
            p_values.append(chi2_p_value)
            cancer1.append(tmp.at[1,"cnt"])
            cancer0.append(tmp.at[0,"cnt"])
        i = i + 1

    tmp = pd.DataFrame({
        'Feature': vars,
        'Benign': cancer0,
        'Malignant': cancer1,
        'chi2 p-value': p_values
        })
    
    if _latex:
        print(tmp.to_latex(),'\n')
    else:
        print(z,'\n')
        print(tmp.to_string(index=False),'\n')
        



# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(utils.image_manipulator)

report_overview(True)

tmp = random_forest_cv(10)



df = pd.read_csv('results/20240223_1150cancer_loss_results.csv', sep=';')
col = [s.strip() for s in list(df.columns)]
df.columns = col

group_columns = ['date', 'img_size', 'target_feature', 'augument', 'model_name', 'epochs', 'filter', 'learning_rate', 
                 'batch_size', 'optimizer', 'loss', 'train_dataset_size', 'test_dataset_size']

res = df.groupby(group_columns).agg({
        'accuracy': ['min', 'max', 'mean'],
        'auc': ['min', 'max', 'mean'],
        'sensitivity': ['min', 'max', 'mean'],
        'specificity': ['min', 'max', 'mean'],
        'precision': ['min', 'max', 'mean'],
        'threshold': ['min', 'max', 'mean'],
        'target_ratio_train': 'mean',
        'target_ratio_test': 'mean',
        'test_positives': 'mean',
        'elapsed_mins': 'sum'
    })

agg_columns = res.columns

res = res.reset_index()
res.columns = group_columns + ['_'.join(col).strip() for col in agg_columns]
res = res.rename(columns={"loss": "loss_function", "date": "run_date", 'elapsed_mins_sum': 'elapsed_mins'})
res['iterations'] = 20
res['status'] = 'OK'

target_cols = ['model_name', 'learning_rate', 'batch_size', 'optimizer', 'loss_function', 'img_size', 'target_feature',
               'augument', 'filter', 'epochs', 'train_dataset_size', 'test_dataset_size', 'target_ratio_train_mean', 
               'test_positives_mean', 'target_ratio_test_mean'	, 'accuracy_min', 'accuracy_max', 'accuracy_mean', 'auc_min',
               'auc_max', 'auc_mean', 'sensitivity_min', 'sensitivity_max', 'sensitivity_mean', 'specificity_min', 
               'specificity_max',	'specificity_mean', 'precision_min', 'precision_max', 'precision_mean',
               'threshold_min', 'threshold_max', 'threshold_mean'	, 'run_date','elapsed_mins', 'iterations', 'status']


res = res[target_cols]

res.to_csv('tmp.csv', sep=';')

