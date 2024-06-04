import os

if os.path.exists("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")

if os.path.exists("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin")

import pandas as pd
import numpy as np
import work.models as m
import work.data as d
import work
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
import yaml    
from datetime import datetime
import utils
import keras
import random
from datetime import timedelta
import importlib
import utils
import utils.custom_logger as cl
import timeit
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

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
    
    CV_ITERATIONS = cfg['CV_ITERATIONS']
    EPOCHS = cfg['EPOCHS']
    SEED = cfg['SEED']
logger = cl.get_logger()


def reinitialize(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))

def generte_model_config(_res_filename):
    
    cfg = get_model_config();
      
    histories = pd.DataFrame(columns =[
        "model_name", "learning_rate", "batch_size", "optimizer", "loss_function", "img_size", 
        "target_feature", "augument", "filter"])
        
    for _, r in cfg.iterrows():
        
        new_row = {
            "model_name": r["model_name"], 
            "learning_rate": r["learning_rate"], 
            "batch_size": r["batch_size"], 
            "optimizer": r["optimizer"], 
            "loss_function": r["loss_function"], 
            "img_size": r["img_size"], 
            "target_feature": "cancer", 
            "augument": 0, 
            "filter": "none"

            }
        
        histories = pd.concat([histories, pd.DataFrame([new_row])], ignore_index=True)
    
    histories.to_csv(_res_filename, mode='w', header=True, index=False, sep=';')
                    
            
    #return histories

def get_model_config():
    
    #learning_rates = [0.01, 0.005]
    learning_rates = [0.005]
    
    #batch_sizes = [8, 16, 32]
    batch_sizes = [16]
        
    #optimizers = ['Adam', 'SGD']
    optimizers = ['SGD'] 

    #losses = ['focal_loss', 'binary_crossentropy', 'squared_hinge', 'categorical_hinge', 'kl_divergence', 'categorical_crossentropy' ]
    losses = ['categorical_crossentropy']
    
    #img_sizes = [80, 100, 120, 140, 160]
    img_sizes = [80]
    
    
    #models = ['cnn1', 'cnn2', 'cnn3', 'VGG16', 'VGG19', 'denseNet121', 'denseNet201']
    models = ['cnn1', 'cnn5',]
    res = pd.DataFrame(columns = ["model_name", "learning_rate", "batch_size", "optimizer", "loss_function", "img_size"])

    for model in models:
        for l in learning_rates:
            for b in batch_sizes:
                for o in optimizers:     
                    for lo in losses:     
                        for img_size in img_sizes:
                            new_row = {'model_name':model, 'learning_rate':l, 'batch_size':b, 'optimizer':o, 'loss_function': lo, 'img_size': img_size}
                            res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)
                        
    return res


def result_found(_df, _c):
    
    founded = _df[
        (_df['loss_function'] == _c['loss_function']) & 
        (_df['model_name'] == _c['model_name']) &
        (_df['learning_rate'] == _c['learning_rate']) &
        (_df['optimizer'] == _c['optimizer']) &
        (_df['img_size'] == _c['img_size']) &
        (_df['batch_size'] == _c['batch_size']) &
        (not _df['status'].isna().any())
        ]
    
    if len(founded)>0: 
        return True
    else:
        return False

def model_cv_iterator(_c, _epochs, _iters):
    
    logger.info('starting model training....')

    
    if _c['filter'] == "none": INPUT_PATH = IMG_PATH_BASE
    if _c['filter'] == "canny": INPUT_PATH = IMG_PATH_CANNY
    if _c['filter'] == "heat": INPUT_PATH = IMG_PATH_HEAT
    if _c['filter'] == "sobel": INPUT_PATH = IMG_PATH_SOBEL
    if _c['filter'] == "bw": INPUT_PATH = IMG_PATH_BW
    if _c['filter'] == "felzen": INPUT_PATH = IMG_PATH_FELZEN
    
    res = pd.DataFrame()
    start = timeit.default_timer()
    iter_num = 0
    
    for i in range(_iters):
        iter_num = iter_num + 1
        logger.info('starting iteration ' + str(iter_num) + ' of ' + str(_iters))
        
        if _c['target_feature']=='cancer':
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4cancer(BASE_FILE_PATH, INPUT_PATH, _c['augument'], 0.15, 0.1, _c['img_size'], SEED )
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4feature(BASE_FILE_PATH, INPUT_PATH, _c['augument'], 0.15, 0.1, _c['feature'], SEED )
                     
        m1 = m.get_model_by_name(_c['model_name'], _c['img_size'], _c['img_size'])
          
        keras.backend.clear_session()
        
        ev = m.model_fitter(m1, X_train, y_train, X_val, y_val, X_test, y_test, _epochs, 
                            _c['learning_rate'], _c['batch_size'], _c['optimizer'], _c['loss_function'], _c['model_name']);
                          
        res = pd.concat([res, pd.DataFrame([ev])], ignore_index=True)    
          
        curr_date = datetime.now().strftime("%Y%m%d_%H%M")
         
        
    stop = timeit.default_timer()
    elapsed = timedelta(minutes=stop-start)
    
    grouped_aggregated = res.agg({
        'accuracy': ['min', 'max', 'mean'],
        'auc': ['min', 'max', 'mean'],
        'sensitivity': ['min', 'max', 'mean'],
        'specificity': ['min', 'max', 'mean'],
        'precision': ['min', 'max', 'mean'],
        'threshold': ['min', 'max', 'mean']
    })
    
    single_row_df = grouped_aggregated.unstack().to_frame().transpose()
    single_row_df.columns = ['_'.join(col).strip() for col in single_row_df.columns.values]


    #grouped_aggregated.columns = ['_'.join(col).strip() for col in grouped_aggregated.columns.values]
    new_row = single_row_df.to_dict('records')[0]
    new_row['elapsed_mins'] = elapsed.seconds//1800
    new_row['train_dataset_size'] = len(y_train)
    new_row['test_dataset_size'] = len(y_test)
    new_row['run_date'] = curr_date
    new_row['epochs'] = _epochs
    new_row['iterations'] = _iters
    new_row['status'] = 'OK'

    new_row.update(_c)
    logger.info(new_row)
    return new_row
        
def train_cv(_config_file, _result_file, _new_result_file, _epochs, _iters):
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
            
    cfg = pd.read_csv(_config_file, sep=";")
    df = pd.read_csv(_result_file, sep=";")
    
    for _, c in cfg.iterrows():
        if not(result_found(df, c)):
            logger.info("processing config ...") 
            logger.info(c.to_dict())
    
            res = model_cv_iterator(c, _epochs, _iters)
            df = pd.concat([df, pd.DataFrame([res])], ignore_index=True)
            df.to_csv(_new_result_file, mode='w', header=True, index=False, sep=';')
            logger.info("results saved ...") 
        else: 
            logger.info("config found! skipping...")
            logger.info(c.to_dict())
            
def main_loop(_config_file, _result_file, _new_result_file, _epochs, _iters):
    logger.info("training loop starting... epochs: " + str(_epochs) + " cv iterations: " + str(_iters))
    train_cv(_config_file, _result_file, _new_result_file, _epochs, _iters)
    logger.info("training finished!")

# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(utils.image_manipulator)

#generte_model_config('config/model_config.csv')

main_loop("config/model_config.csv", 'results/results6.csv', 'results/results7.csv', 100, 10)

