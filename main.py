import pandas as pd
import numpy as np
import work.models as m
import work.data as d
import work
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
import yaml    
import logging
from datetime import datetime
import utils
import keras
import random
from datetime import timedelta
import importlib

import timeit

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

TRAIN_PATH = BASE_PATH + 'modeling/all_images/train/'
VAL_PATH = BASE_PATH + 'modeling/all_images/validation/'
TEST_PATH = BASE_PATH + 'modeling/all_images/test/'

TRAIN_PATH_BW = BASE_PATH + 'modeling/all_images_bw/train/'
TEST_PATH_BW = BASE_PATH + 'modeling/all_images_bw/test/'
VAL_PATH_BW = BASE_PATH + 'modeling/all_images/bw/validation/'

TRAIN_PATH_SOBEL = BASE_PATH + 'modeling/all_images_sobel/train/'
TEST_PATH_SOBEL = BASE_PATH + 'modeling/all_images_sobel/test/'
VAL_PATH_SOBEL = BASE_PATH + 'modeling/all_images/sobel/validation/'

TRAIN_PATH_HEAT = BASE_PATH + 'modeling/all_images_heat/train/'
TEST_PATH_HEAT = BASE_PATH + 'modeling/all_images_heat/test/'
VAL_PATH_HEAT = BASE_PATH + 'modeling/all_images/heat/validation/'

TRAIN_PATH_CANNY = BASE_PATH + 'modeling/all_images_canny/train/'
TEST_PATH_CANNY = BASE_PATH + 'modeling/all_images_canny/test/'
VAL_PATH_CANNY = BASE_PATH + 'modeling/all_images/canny/validation/'

TRAIN_PATH_FELZEN = BASE_PATH + 'modeling/all_images_felzen/train/'
TEST_PATH_FELZEN = BASE_PATH + 'modeling/all_images_felzen/test/'
VAL_PATH_FELZEN = BASE_PATH + 'modeling/all_images/felzen/validation/'

BATCH_SIZE = 12
EPOCHS = 100
BASE_LR = 0.001#1e-6
VERBOSE_FLAG = False


def train_single_model(_epochs):
    INPUT_PATH = BASE_PATH + 'modeling/all_images/'
    OUTPUT_TEST_PATH = TEST_PATH
    OUTPUT_TRAIN_PATH = TRAIN_PATH
    OUTPUT_VAL_PATH = VAL_PATH
    X_train, y_train, X_val, y_val, X_test, y_test = d.split_data(BASE_FILE_PATH, OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH, OUTPUT_TEST_PATH, 0)
        
    m1 = m.model_cnn1(IMG_WIDTH, IMG_HEIGHT)
    
    m1.compile(optimizer = Adam(learning_rate = BASE_LR), loss = 'categorical_crossentropy', metrics=["accuracy"]) 
    hist = m1.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=_epochs)
    
    ev = m1.evaluate(X_test, y_test)
    
    logger.info("processing finished! accuracy=" + str(round(ev[1], 2)));
    
    return m1
     
def reinitialize(model):
    for l in model.layers:
        if hasattr(l,"kernel_initializer"):
            l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
        if hasattr(l,"bias_initializer"):
            l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
        if hasattr(l,"recurrent_initializer"):
            l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
            
def get_config():
    # res = pd.DataFrame(columns = ['learning_rate', 'batch_size', 'optimizer'])
    # learning_rates = [0.05, 0.01, 0.005]
    # batch_sizes = [8, 16, 32]
    # optimizers = ['Adam', 'SDG']
    res = pd.DataFrame(columns = ['learning_rate', 'batch_size', 'optimizer'])
    learning_rates = [0.005]
    batch_sizes = [32]
    optimizers = ['SDG'] 
    for l in learning_rates:
        for b in batch_sizes:
            for o in optimizers:        
                new_row = {'learning_rate':l, 'batch_size':b, 'optimizer':o}
                #res = res._append(new_row, ignore_index=True)
                res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)
                
    return res

            
def train_model_multi_cv(_epochs, _iters, _filter="none", _feature="cancer"):
    
    random.seed(123)
    np.random.seed(123)
    tf.keras.utils.set_random_seed(123)
    
    
    if _filter == "none": 
        INPUT_PATH = BASE_PATH + 'modeling/all_images/'
        OUTPUT_TEST_PATH = TEST_PATH
        OUTPUT_TRAIN_PATH = TRAIN_PATH
        OUTPUT_VAL_PATH = VAL_PATH
    if _filter == "canny":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_canny/'
        OUTPUT_TEST_PATH = TEST_PATH_CANNY
        OUTPUT_TRAIN_PATH = TRAIN_PATH_CANNY
        OUTPUT_VAL_PATH = VAL_PATH_CANNY
    if _filter == "heat":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_heat/'
        OUTPUT_TEST_PATH = TEST_PATH_HEAT
        OUTPUT_TRAIN_PATH = TRAIN_PATH_HEAT
        OUTPUT_VAL_PATH = VAL_PATH_HEAT
    if _filter == "sobel":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_sobel/'
        OUTPUT_TEST_PATH = TEST_PATH_SOBEL
        OUTPUT_TRAIN_PATH = TRAIN_PATH_SOBEL
        OUTPUT_VAL_PATH = VAL_PATH_SOBEL
    if _filter == "bw":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_bw/'
        OUTPUT_TEST_PATH = TEST_PATH_BW
        OUTPUT_TRAIN_PATH = TRAIN_PATH_BW
        OUTPUT_VAL_PATH = VAL_PATH_BW
    if _filter == "felzen":
        INPUT_PATH = BASE_PATH + 'modeling/all_images_felzen/'
        OUTPUT_TEST_PATH = TEST_PATH_FELZEN
        OUTPUT_TRAIN_PATH = TRAIN_PATH_FELZEN
        OUTPUT_VAL_PATH = VAL_PATH_FELZEN

    
    _, models = m.model_sequence_manual_1(IMG_WIDTH, IMG_HEIGHT)
    model_cnt = len(models)
    
    logger.info('processing start... ' + 'models: ' + str(model_cnt) + ' cv iters: ' + str(_iters) + ' filter: ' + str(_filter))
    
    config = get_config()
    
    total_runs = _iters * len(config) * model_cnt         
    run_num = 0
    for i in range(_iters):
        
        d.split_files(INPUT_PATH, OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH, OUTPUT_TEST_PATH, 0.15, 0.1)
         
        if _feature=='cancer':
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4cancer(BASE_FILE_PATH, OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH, OUTPUT_TEST_PATH, 0)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4feature(BASE_FILE_PATH, OUTPUT_TRAIN_PATH, OUTPUT_VAL_PATH, OUTPUT_TEST_PATH, 0, _feature)
            
        for idx, c in config.iterrows():
            
            for m_num in range(model_cnt):
                run_num = run_num + 1
                start = timeit.default_timer()
                names, models = m.model_sequence_manual_1(IMG_WIDTH, IMG_HEIGHT)
                m1 = models[m_num]
                m1_name = names[m_num]
                
                keras.backend.clear_session()
                
                model_name = "models/" + m1_name +"_" + _feature  + "_" + _filter + "_" + str(run_num)
                ev = m.model_fitter(m1, X_train, y_train, X_val, y_val, X_test, y_test, _epochs, c['learning_rate'], c['batch_size'], c['optimizer'], model_name);
                            
                
                stop = timeit.default_timer()

                elapsed = timedelta(minutes=stop-start)

                histories = pd.DataFrame(columns =["date", "target_feature", "run_num", "total_runs", "model_name", "model_num", "iter_num", "filter",  "target_ratio_train", 
                                                   "target_ratio_test","accuracy", "train_dataset_size", "test_dataset_size",
                                                   "learning_rate", "batch_size", "optimizer", "elapsed_mins"])
                

                curr_date = datetime.now().strftime("%Y%m%d_%H%M")
   
                new_row = {'date': curr_date,
                           'target_feature': _feature,
                           'run_num': run_num,
                           'total_runs': total_runs,
                           'model_name': m1_name,
                           'model_num':m_num+1,
                           'iter_num':i+1,
                           'filter':_filter, 
                           'target_ratio_train':round(sum(y_train[:,1])/len(y_train[:,1]),2),
                           'target_ratio_test':round(sum(y_test[:,1])/len(y_test[:,1]),2),
                           'accuracy': round(ev[1], 2),
                           'train_dataset_size':len(y_test),
                           'test_dataset_size':len(y_train),
                           'learning_rate':c['learning_rate'],
                           'batch_size':c['batch_size'],
                           'optimizer':c['optimizer'],
                           'elapsed_mins': elapsed.seconds//1800}
    
                #histories = histories._append(new_row, ignore_index=True)
                histories = pd.concat([histories, pd.DataFrame([new_row])], ignore_index=True)
                histories.to_csv('results.csv', mode='a', header=False, index=False)
                         
                logger.info(new_row)
            
        
    return 1


def main_loop(_epochs, _iters):
    logger.info("starting...")
    

    
    # f = open('results.csv','w') 
    # f.write("date, target_feature, run_num, total_runs, model_name, model_num, iter_num, filter,  target_ratio_train, target_ratio_test, accuracy, train_dataset_size, test_dataset_size, learning_rate, batch_size, optimizer, elapsed_mins\n")
    # f.close()
    

    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'granice_rowne')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'ksztalt_nieregularny')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'Zwapnienia_mikrozwapnienia')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'granice_zatarte')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'echo_gleboko_hipo')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'USG_AZT')    
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'Zwapnienia_makrozwapnienia')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'echo_nieznacznie_hipo')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'brzegi_spikularne')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'ksztalt_owalny')
    hist = train_model_multi_cv(_epochs, _iters, 'heat', 'torbka_modelowanie')    
    
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'granice_rowne')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'ksztalt_nieregularny')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'Zwapnienia_mikrozwapnienia')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'granice_zatarte')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'echo_gleboko_hipo')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'USG_AZT')    
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'Zwapnienia_makrozwapnienia')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'echo_nieznacznie_hipo')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'brzegi_spikularne')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'ksztalt_owalny')
    hist = train_model_multi_cv(_epochs, _iters, 'bw', 'torbka_modelowanie')   

    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'granice_rowne')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'ksztalt_nieregularny')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'Zwapnienia_mikrozwapnienia')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'granice_zatarte')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'echo_gleboko_hipo')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'USG_AZT')    
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'Zwapnienia_makrozwapnienia')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'echo_nieznacznie_hipo')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'brzegi_spikularne')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'ksztalt_owalny')
    hist = train_model_multi_cv(_epochs, _iters, 'sobel', 'torbka_modelowanie')   
    
    logger.info("training finished!")
   
def train_and_save(_epochs, _out_filename):
    logger.info("starting...")
    
    random.seed(123)
    np.random.seed(123)
    tf.keras.utils.set_random_seed(123)
    
    m1 = train_single_model(_epochs)
    m1.save(_out_filename, save_format='tf')
    logger.info("training finished!")
    
    return m1

    

# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(utils.image_manipulator)

main_loop(20,5)



