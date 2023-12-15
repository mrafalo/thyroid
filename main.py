import os
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin")

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
    
    IMG_WIDTH = cfg['IMG_WIDTH']
    IMG_HEIGHT = cfg['IMG_HEIGHT']



logger = cl.get_logger()
BATCH_SIZE = 12
BASE_LR = 0.001#1e-6
SEED = 123


def reinitialize(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
            
def get_model_config():
    # res = pd.DataFrame(columns = ['learning_rate', 'batch_size', 'optimizer'])
    #learning_rates = [0.01, 0.005]
    # batch_sizes = [8, 16, 32]
    # optimizers = ['Adam', 'SDG']
    res = pd.DataFrame(columns = ['learning_rate', 'batch_size', 'optimizer'])
    learning_rates = [0.005]
    batch_sizes = [16]
    optimizers = ['Adam'] 
    for l in learning_rates:
        for b in batch_sizes:
            for o in optimizers:        
                new_row = {'learning_rate':l, 'batch_size':b, 'optimizer':o}
                res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)
                
    return res

            
def train_model_multi_cv(_prefix, _epochs, _iters, _filter="none", _feature="cancer", _augument=0):
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    
    if _filter == "none": INPUT_PATH = IMG_PATH_BASE
    if _filter == "canny": INPUT_PATH = IMG_PATH_CANNY
    if _filter == "heat": INPUT_PATH = IMG_PATH_HEAT
    if _filter == "sobel": INPUT_PATH = IMG_PATH_SOBEL
    if _filter == "bw": INPUT_PATH = IMG_PATH_BW
    if _filter == "felzen": INPUT_PATH = IMG_PATH_FELZEN

    _, models = m.model_sequence_manual_2(IMG_WIDTH, IMG_HEIGHT)
    model_cnt = len(models)
    
    logger.info('processing multiple models start... ' + 'models: ' + str(model_cnt) + 
                ' cv iters: ' + str(_iters) + 
                ' filter: ' + str(_filter) + 
                ' feature: ' + _feature)
    
    config = get_model_config()
    
    total_runs = _iters * len(config) * model_cnt         
    run_num = 0
    for i in range(_iters):
        
        if _feature=='cancer':
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4cancer(BASE_FILE_PATH, INPUT_PATH, _augument, 0.15, 0.1, SEED )
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4feature(BASE_FILE_PATH, INPUT_PATH, _augument, 0.15, 0.1, _feature, SEED )
            
        for idx, c in config.iterrows():
            
            for m_num in range(model_cnt):
                run_num = run_num + 1
                start = timeit.default_timer()
                names, models = m.model_sequence_manual_2(IMG_WIDTH, IMG_HEIGHT)
                m1 = models[m_num]
                m1_name = names[m_num]
                
                keras.backend.clear_session()
                
                model_name = "models/" + m1_name +"_" + _feature  + "_" + _filter + "_" + str(run_num)
                ev = m.model_fitter(m1, X_train, y_train, X_val, y_val, X_test, y_test, _epochs, c['learning_rate'], c['batch_size'], c['optimizer'], model_name);
                            
                
                stop = timeit.default_timer()

                elapsed = timedelta(minutes=stop-start)

                histories = pd.DataFrame(columns =["date", "target_feature", "augument", "run_num", "total_runs", 
                                                   "model_name", "model_num", "iter_num", "filter",  "target_ratio_train", 
                                                   "target_ratio_test","accuracy", "auc", "sensitivity", "specificity",
                                                   "precision", "threshold", "train_dataset_size", "test_dataset_size",
                                                   "learning_rate", "batch_size", "optimizer", 'test_cases', 'test_positives', "elapsed_mins"])
                

                curr_date = datetime.now().strftime("%Y%m%d_%H%M")
   
                new_row = {'date': curr_date,
                           'target_feature': _feature,
                           'augument': _augument,
                           'run_num': run_num,
                           'total_runs': total_runs,
                           'model_name': m1_name,
                           'model_num':m_num+1,
                           'iter_num':i+1,
                           'filter':_filter, 
                           'target_ratio_train':round(sum(y_train[:,1])/len(y_train[:,1]),2),
                           'target_ratio_test':round(sum(y_test[:,1])/len(y_test[:,1]),2),
                           'accuracy': ev['accuracy'],
                           'auc': ev['auc'],
                           'sensitivity': ev['sensitivity'],
                           'specificity': ev['specificity'],
                           'precision': ev['precision'],
                           'threshold': ev['threshold'],
                           'train_dataset_size':len(y_train),
                           'test_dataset_size':len(y_test),
                           'learning_rate':c['learning_rate'],
                           'batch_size':c['batch_size'],
                           'optimizer':c['optimizer'],
                           'test_cases':ev['test_cases'],
                           'test_positives':ev['test_positives'],
                           'elapsed_mins': elapsed.seconds//1800}
    
                histories = pd.concat([histories, pd.DataFrame([new_row])], ignore_index=True)
                histories.to_csv('results/'+_prefix+'_results.csv', mode='a', header=False, index=False)
                         
                logger.info(new_row)
            
        
    return 1


def main_loop(_prefix, _epochs, _iters):
    logger.info("starting...")
    
    f = open('results/'+_prefix+'_results.csv','w') 
    f.write("date, target_feature, augument, run_num, total_runs, model_name, model_num, iter_num, filter,  target_ratio_train, target_ratio_test, accuracy, auc, sensitivity, specificity, precision, threshold, train_dataset_size, test_dataset_size, learning_rate, batch_size, optimizer, test_cases, test_positives, elapsed_mins\n")
    f.close()
    
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'none', 'cancer', 0)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'none', 'cancer', 1)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'heat', 'cancer', 0)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'heat', 'cancer', 1)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'canny', 'cancer',0)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'canny', 'cancer',1)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'bw', 'cancer', 0)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'bw', 'cancer', 1)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'sobel', 'cancer', 0)
    hist = train_model_multi_cv(_prefix, _epochs, _iters, 'sobel', 'cancer', 1)
    # hist = train_model_multi_cv(_epochs, _iters, 'heat', 'ksztalt_nieregularny')
    # hist = train_model_multi_cv(_epochs, _iters, 'heat', 'Zwapnienia_mikrozwapnienia')
    # hist = train_model_multi_cv(_epochs, _iters, 'heat', 'granice_zatarte')
    # hist = train_model_multi_cv(_epochs, _iters, 'heat', 'echo_gleboko_hipo')
    # hist = train_model_multi_cv(_epochs, _iters, 'heat', 'USG_AZT')    
    # hist = train_model_multi_cv(_epochs, _iters, 'heat', 'Zwapnienia_makrozwapnienia')
    # hist = train_model_multi_cv(_epochs, _iters, 'heat', 'echo_nieznacznie_hipo')
    # hist = train_model_multi_cv(_epochs, _iters, 'heat', 'torbka_modelowanie')    
 
    
    logger.info("training finished!")

# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(utils.image_manipulator)

main_loop("cancer_filters",1, 1)
