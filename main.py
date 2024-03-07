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

<<<<<<< HEAD
logger = cl.get_logger()

SEED = 123

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
    
        # histories = pd.DataFrame(columns =[
        # "model_name", "learning_rate", "batch_size", "optimizer", "loss_function", "img_size", 
        # "target_feature", "augument", "filter", "iterations", "epochs",
        # "target_ratio_train", "target_ratio_test", "train_dataset_size", "test_dataset_size", "test_cases", "test_positives", 
        # "accuracy_min", "accuracy_max", "accuracy_mean", 
        # "auc_min", "auc_max", "auc_mean", 
        # "sensitivity_min", "sensitivity_max", "sensitivity_mean", 
        # "specificity_min", "specificity_max", "specificity_mean",
        # "precision_min", "precision_max", "precision_mean", 
        # "threshold_min", "threshold_max", "threshold_mean",
        # "run_date", "elapsed_mins", "status"])
        
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
                    
            
    return histories

def get_model_config():
=======
def train_models(_X, _y):
>>>>>>> eea28a252afd4e7e9564e8c7e1c5d17277f6721b
    
    res = pd.DataFrame(columns = ['model', 'mse', 'mape', 'r2'])
    
<<<<<<< HEAD
    #batch_sizes = [8, 16, 32]
    batch_sizes = [16]
        
    optimizers = ['Adam', 'SGD']
    #optimizers = ['Adam'] 

    losses = ['focal_loss', 'binary_crossentropy', 'squared_hinge', 'categorical_hinge', 'kl_divergence', 'categorical_crossentropy' ]
    #losses = ['kl_divergence']
    
    img_sizes = [120, 180]
    
    models = ['cnn1', 'cnn2', 'cnn3', 'VGG16', 'VGG19', 'denseNet121', 'denseNet201']
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


def get_model_config_old():
    
    #learning_rates = [0.01, 0.005]
    learning_rates = [0.005]
    
    #batch_sizes = [8, 16, 32]
    batch_sizes = [16]
        
    optimizers = ['Adam', 'SDG']
    #optimizers = ['Adam'] 
=======
    df_preds = pd.DataFrame({'y_actual': _y})
    res = pd.DataFrame(columns = ['model', 'mse', 'mape', 'r2'])
>>>>>>> eea28a252afd4e7e9564e8c7e1c5d17277f6721b

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
        
<<<<<<< HEAD
    return 1

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
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4cancer(BASE_FILE_PATH, INPUT_PATH, _c['augument'], 0.15, 0.1, SEED )
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = d.split_data_4feature(BASE_FILE_PATH, INPUT_PATH, _c['augument'], 0.15, 0.1, _c['feature'], SEED )
                     
        m1 = m.get_model_by_name(_c['model_name'], IMG_WIDTH, IMG_HEIGHT)
          
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
        
def train_cv(_config_file, _result_file, _epochs, _iters):
    
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
            df.to_csv("results/tmp.csv", mode='w', header=True, index=False, sep=';')
            logger.info("results saved ...") 
        else: 
            logger.info("config found! skipping...")
            logger.info(c.to_dict())
            
def main_loop(_config_file, _result_file, _epochs, _iters):
    logger.info("training loop starting... epochs: " + str(_epochs) + " cv iterations: " + str(_iters))
    train_cv(_config_file, _result_file, _epochs, _iters)
    logger.info("training finished!")
=======
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
>>>>>>> eea28a252afd4e7e9564e8c7e1c5d17277f6721b

# importlib.reload(work.models)
# importlib.reload(work.data)

<<<<<<< HEAD
#generte_model_config('config/config.csv')


main_loop("config/config.csv", 'config/results5.csv', 1, 2)

=======

main_loop()
>>>>>>> eea28a252afd4e7e9564e8c7e1c5d17277f6721b
