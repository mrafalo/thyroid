import os
if os.path.exists("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")

if os.path.exists("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin")
    
import numpy as np
import pandas as pd
import work
import work.models as m
import work.data as d
import yaml    
import importlib
from datetime import datetime
import tensorflow as tf
import random
import utils
import utils.custom_logger as cl
from sklearn.decomposition import FastICA
import glob
import time

logger = cl.get_logger()

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    TELCO_FILE = cfg['TELCO_FILE']
    SEED = cfg['SEED']
    EPOCHS = cfg['EPOCHS']
    SAMPLE_SIZE = cfg['SAMPLE_SIZE']
    ITERATIONS = cfg['ITERATIONS']
    MODEL_CONFIG_FILE = cfg['MODEL_CONFIG_FILE']

def train_models(_X, _y):
    
    res = pd.DataFrame(columns = ['model', 'mse', 'mape', 'r2'])
    
    df_preds = pd.DataFrame({'y_actual': _y})
    res = pd.DataFrame(columns = ['model', 'mse', 'mape', 'r2'])

    model_cfg = pd.read_csv(MODEL_CONFIG_FILE, sep=";")
    for _,c in model_cfg.iterrows():
        logger.info("model " + c['model_name'] + " training start dataset size: " + str(len(_X)))
        
        sizes = [int(num) for num in c['sizes'].split(',')]
        activations = [ss for ss in c['activations'].split(',')]
        m1 = m.model_nn_custom(c['layers'], sizes, activations, len(_X.columns))
        
        mse, mape, r2, preds = m.model_fiter(m1, _X, _y, _X, _y, False)
    
        new_row = {'model':c['model_name'], 'mse':mse, 'mape':mape, 'r2': r2}
        res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)    
        df_preds[c['model_name']] = preds
        
        logger.info("model " + c['model_name'] + " training finished...")
                
    return df_preds, res

def ica_results(_mask):
    _mask = 'ica_res*.csv'
    path_pattern = 'results/' + _mask
    ica_files = glob.glob(path_pattern)
    
    for f in ica_files:
        df = pd.read_csv(f, sep=';')
        model_columns = [x for x in list(df.columns) if x not in ('scenario', 'predictions_file')]
        number_of_components = len(model_columns)
        
        base = df.loc[df.scenario == 'mse_base',model_columns].values
        components = df.loc[df.scenario != 'mse_base',model_columns].values
        
        for i in range(number_of_components):
            mse_reduction_prc = (components[i,:] - base) / base
            
            if np.mean(mse_reduction_prc) < 0:
                print(f, np.mean(mse_reduction_prc))
        

def ica_iterator(_predictions_file, _iter):
    
    df = pd.read_csv('results/' + _predictions_file, sep=';')

    y_actual = df['y_actual'].values
    
    x = df.drop('y_actual', axis=1).values
    components = x.shape[1]
    
    ica = FastICA()
    y = ica.fit_transform(x) 
    mses = np.zeros((components + 1,components))
    
    scenarios = [] 
    scenarios.append('mse_base')
    
    for j in range(components):
        mse = m.mse_score(y_actual, x.T[j])
        print(mse)
        mses[0,j] = round(mse)

    for i in range(components):
        z = y.copy()
        z[:,i] = 0
        scenarios.append('mse_s_' + str(i))
        
        xp = ica.inverse_transform(z)
        for j in range(components):
            mse = m.mse_score(y_actual, xp[:,j])
            mses[i+1,j] = round(mse)
    
    res = pd.DataFrame(mses, columns=df.drop('y_actual', axis=1).columns)
    tmp = pd.DataFrame(scenarios, columns=['scenario'])   
    res_final = pd.concat([tmp, res], ignore_index=True, axis=1)
    res_final.columns =  list(tmp.columns)+list(res.columns)
    res_final['predictions_file'] = _predictions_file
    curr_date = datetime.now().strftime("%Y%m%d_%H%M")
    
    res_final.to_csv("results/ica_result_" + str(_iter) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")
    
def main_loop():
    
    logger.info("starting... epochs: " + str(EPOCHS))
     
    np.random.seed(m.SEED)
    tf.keras.utils.set_random_seed(m.SEED)
    random.seed(SEED)
    start = time.time()

    for i in range(ITERATIONS):
        logger.info("starting... iteration: " + str(i+1) + "/" + str(ITERATIONS))
        
        X_train, y_train, X_test, y_test = d.get_data();
    
        res_preds, res_summary = train_models(X_train, y_train)
    
        curr_date = datetime.now().strftime("%Y%m%d_%H%M")
        predictions_file = "models_predictions_" + str(i) + " _" + curr_date + ".csv"
        res_preds.to_csv("results/" + predictions_file, mode='w', header=True, index=False, sep=";")
        res_summary.to_csv("results/models_summary_" + str(i+1) + " _" + curr_date + ".csv", mode='w', header=True, index=False, sep=";")
        
        ica_iterator(predictions_file, i+1)
        logger.info("done... iteration: " + str(i+1) + "/" + str(ITERATIONS))
        
    stop = time.time()
    
    elapsed_sec = stop-start
    logger.info("training finished!, elapsed: " + str(elapsed_sec//60) + " minutes")

# importlib.reload(work.models)
# importlib.reload(work.data)


main_loop()
