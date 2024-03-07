
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
from sklearn.decomposition import FastICA
from keras.models import Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    TELCO_FILE = cfg['TELCO_FILE']
    SEED = cfg['SEED']
    EPOCHS = cfg['EPOCHS']
    


def compute_D(_y, _z, _beta):
 
    if _beta >0:
        res = sum((_y * (pow(_y, _beta) - pow(_z, _beta)) / _beta) - (_y * (pow(_y, _beta+1) - pow(_z, _beta+1)) / (_beta+1)))
        return res
    
    if _beta == 0:
        res = sum(_y * np.log(_y/_z) - _y + _z)
        return res
 
    if _beta == -1:
        res = sum(np.log(_z/_y) - _y/_z - 1)
        return res
    
def compute_J(_y):

    x = np.array([1,2,3,4,5,6])
    y = np.array([2,3,5,7,8,1])
    
    compute_D(x, y, -1)
    
    return 1
    
    
def ica_model(_X):
    ica2 = FastICA()
    y = ica2.fit_transform(_X) 
    #w = ica.mixing_
    #w = ica.components_
    return y

def mape_score(_y_test, _y_pred):
    
<<<<<<< HEAD
    return model

def model_sequence_auto(_img_width, _img_height):
    models = []
    conv_filter_sizes = [(2,2), (3,3)]    
    pool_filter_sizes = [(1,1), (2,2), (3,3)]   
    conv_sizes = [2, 4, 6]
    conv_filters = [8, 16, 24, 32, 48]
    
    for conv_filter in conv_filters:
        for conv_size in conv_sizes:
            for conv_filter_size in conv_filter_sizes:
                for pool_filter_size in pool_filter_sizes:
                
                    model = Sequential()
                    model.add(Input(shape=(_img_width, _img_height, 1)))
                    model._name = "conv_filter_{}_conv_size_{}_conv_filter_size_{}_pool_filter_size_{}".format(conv_filter, conv_size, conv_filter_size[0], pool_filter_size[0])
                    #model._name = "zigi_12"
                    
                    for i in range(1,conv_size):
                        #print(conv_size, conv_filter_size, pool_filter_size, i)
                        model.add(Conv2D(conv_filter, conv_filter_size, padding="same", activation="relu"))
                        model.add(MaxPool2D(pool_filter_size, strides=(2, 2)))
                    
                    model.add(Flatten())
                    model.add(Dense(2, activation='softmax'))
                    
                    models.append(model)
        
    print(len(models), "models prepared...")
    return models


def model_sequence_manual_1(_img_width, _img_height):
    models = []
    names = ['cnn1']
               
    models.append(model_cnn1(_img_width, _img_height))
    # models.append(model_cnn4(_img_width, _img_height))
    # models.append(model_densenet201(_img_width, _img_height))
        
    return names, models

def get_model_by_name(_model_name,_img_width, _img_height ):
    return model_cnn1(_img_width, _img_height)
    
def model_sequence_manual_2(_img_width, _img_height):
    models = []
    names = ['VGG16', 'VGG19', 'denseNet201', 
             'denseNet121', 'cnn1', 'cnn2', 'cnn3', 
             'cnn4', 'cnn5', 'cnn6', 'cnn7', "ResNet50", "ResNet101", "ResNet152"]
               

    models.append(model_VGG16(_img_width, _img_height))
    models.append(model_VGG19(_img_width, _img_height))
    models.append(model_densenet201(_img_width, _img_height))
    models.append(model_densenet121(_img_width, _img_height))
    models.append(model_cnn1(_img_width, _img_height))
    models.append(model_cnn2(_img_width, _img_height))
    models.append(model_cnn3(_img_width, _img_height))
    models.append(model_cnn4(_img_width, _img_height))
    models.append(model_cnn5(_img_width, _img_height))
    models.append(model_cnn6(_img_width, _img_height))
    models.append(model_cnn7(_img_width, _img_height))      
    models.append(model_ResNet50(_img_width, _img_height))
    models.append(model_ResNet101(_img_width, _img_height))  
    models.append(model_ResNet152(_img_width, _img_height))      
=======
    return sum(abs((_y_test - _y_pred) / _y_test)) / len(_y_test);
>>>>>>> eea28a252afd4e7e9564e8c7e1c5d17277f6721b

def mse_score(_y_test, _y_pred):
    
    return sum(pow(_y_test - _y_pred,2)) / len(_y_test);

def model_random_forest(_X_train, _y_train, _X_test, _y_test):
    m1 = RandomForestRegressor(n_estimators=10)
    m1 = m1.fit(_X_train, _y_train)
    mse, mape, r2, preds = model_preditor(m1, _X_test, _y_test)
    return mse, mape, r2, preds

<<<<<<< HEAD

def model_sequence_manual_3(_img_width, _img_height):
    models = []
    names = ["ResNet50", "ResNet101", "ResNet152"]
               
    models.append(model_ResNet50(_img_width, _img_height))
    models.append(model_ResNet101(_img_width, _img_height))  
    models.append(model_ResNet152(_img_width, _img_height))  
        
    return names, models  

def model_sequence_manual_ALL(_img_width, _img_height):
    models = []
    names = ['VGG16', 'VGG19', 'denseNet201', 
             'denseNet121', 'cnn1', 'cnn2', 'cnn3', 
             'cnn4', 'cnn5', 'cnn6', 'cnn7', "ResNet50", "ResNet101", "ResNet152"]
               

    models.append(model_VGG16(_img_width, _img_height))
    models.append(model_VGG19(_img_width, _img_height))
    models.append(model_densenet201(_img_width, _img_height))
    models.append(model_densenet121(_img_width, _img_height))
    models.append(model_cnn1(_img_width, _img_height))
    models.append(model_cnn2(_img_width, _img_height))
    models.append(model_cnn3(_img_width, _img_height))
    models.append(model_cnn4(_img_width, _img_height))
    models.append(model_cnn5(_img_width, _img_height))
    models.append(model_cnn6(_img_width, _img_height))
    models.append(model_cnn7(_img_width, _img_height))      
    models.append(model_ResNet50(_img_width, _img_height))
    models.append(model_ResNet101(_img_width, _img_height))  
    models.append(model_ResNet152(_img_width, _img_height))      

    
    return names, models

def model_densenet201(_img_width, _img_height):
=======

def model_nn_custom(_layers, _sizes, _activations, _input_size):
>>>>>>> eea28a252afd4e7e9564e8c7e1c5d17277f6721b
    
    model = Sequential()
    model.add(Input(shape=(_input_size,)))
    
    for i in range(_layers):
        model.add(Dense(_sizes[i], activation=_activations[i]))
    
    model.add(Dense(1, activation='linear'))
    
    return model



def model_nn_1(_X_train, _y_train, _X_test, _y_test):
    m1 = keras.Sequential([
        layers.Input(shape=(len(_X_train.columns),)),       
        layers.Dense(256, activation='relu'),  
        layers.Dense(128, activation='relu'),  
        layers.Dense(64, activation='relu'),  
        layers.Dense(1, activation='linear')                  
    ])
    m1.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    
    
    m1.fit(_X_train, _y_train, epochs=EPOCHS, batch_size=32, validation_data=(_X_test, _y_test),  callbacks=[es], verbose=False)
    
    mse, mape, r2, preds = model_preditor(m1, _X_test, _y_test)
    return mse, mape, r2, preds

def model_nn_2(_X_train, _y_train, _X_test, _y_test):
    m1 = keras.Sequential([
        layers.Input(shape=(len(_X_train.columns),)),       
        layers.Dense(128, activation='relu'),  
        layers.Dense(64, activation='relu'),  
        layers.Dense(32, activation='relu'),  
        layers.Dense(1, activation='linear')                  
    ])
    m1.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    
    
    m1.fit(_X_train, _y_train, epochs=EPOCHS, batch_size=32, validation_data=(_X_test, _y_test),  callbacks=[es], verbose=False)
    
    mse, mape, r2, preds = model_preditor(m1, _X_test, _y_test)
    return mse, mape, r2, preds

def model_nn_3(_X_train, _y_train, _X_test, _y_test):
    m1 = keras.Sequential([
        layers.Input(shape=(len(_X_train.columns),)),       
        layers.Dense(128, activation='relu'),  
        layers.Dense(128, activation='relu'),  
        layers.Dense(128, activation='relu'),  
        layers.Dense(1, activation='linear')                  
    ])
    m1.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    
    
    m1.fit(_X_train, _y_train, epochs=EPOCHS, batch_size=32, validation_data=(_X_test, _y_test),  callbacks=[es], verbose=False)
    
    mse, mape, r2, preds = model_preditor(m1, _X_test, _y_test)
    return mse, mape, r2, preds


    
# def model_xgb(_X_train, _y_train, _X_test, _y_test):
#     m1 = XGBRegressor()
#     m1 = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
#     m1.fit(_X_train, _y_train)
#     mse, mape, r2, preds = model_preditor(m1, _X_test, _y_test)
#     return mse, mape, r2, preds

def model_fiter(_model,_X_train, _y_train, _X_test, _y_test, _scale=False):
    
    if _scale:
        sc = StandardScaler()
        _X_train = sc.fit_transform(_X_train)
        _X_test = sc.transform(_X_test)
        scale = 'scaled'
    else:
        scale = 'not_scaled'
            
        
    _model.compile(optimizer='adam', loss='mean_squared_error')
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    
    _model.fit(_X_train, _y_train, epochs=EPOCHS, batch_size=32, validation_data=(_X_test, _y_test),  callbacks=[es], verbose=False)
    
    mse, mape, r2, preds = model_preditor(_model, _X_test, _y_test)
    return mse, mape, r2, preds


def model_preditor(_model, _X_test, _y_test):
    m1_predict = _model.predict(_X_test, verbose=False)
    
    if len(m1_predict.shape) > 1:
        m1_predict = m1_predict[:,0]
    
    m1_mse = round(mean_squared_error(_y_test, m1_predict),3)
    m1_mape = round(mape_score(_y_test, m1_predict),3)
    m1_r2 = round(r2_score(_y_test, m1_predict), 3)

<<<<<<< HEAD
def model_predictor_scikit(_model, _X_test, _y_test):
    
    y_base = _y_test

    test_cases = len(_y_test)
    test_positives = np.sum(_y_test)
    
    if test_positives==0:
        return {
        'accuracy': -1,
        'sensitivity': -1,
        'specificity': -1,
        'precision': -1,
        'f1': -1,
        'auc': -1,
        'threshold': -1,
        'test_cases': test_cases,
        'test_positives': test_positives
        }
    
    y_predict_base = _model.predict_proba(_X_test)
    
    m_opt_predict = y_predict_base[:,1]
    
    t = find_cutoff(_y_test,m_opt_predict)

    m_opt_predict_binary = [1 if x >= t else 0 for x in m_opt_predict]
 
    conf_matrix = np.round(metrics.confusion_matrix(_y_test, m_opt_predict_binary),2)
    
   
    accuracy = np.round(metrics.accuracy_score(_y_test, m_opt_predict_binary),2)
    sensitivity = np.round(metrics.recall_score(_y_test, m_opt_predict_binary),2)
    specificity = np.round(conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),2)
    precision = np.round(metrics.precision_score(_y_test, m_opt_predict_binary),2)
    auc = np.round(metrics.roc_auc_score(_y_test, m_opt_predict),2)
    f1 = np.round(metrics.f1_score(_y_test, m_opt_predict_binary),2)
    test_cases = len(_y_test)
    test_positives = np.sum(_y_test)
    
    res = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'threshold': t,
        'test_cases': test_cases,
        'test_positives': test_positives
    }
    
    return res

def model_load(_path):
=======
    return m1_mse, m1_mape, m1_r2, m1_predict
>>>>>>> eea28a252afd4e7e9564e8c7e1c5d17277f6721b

    
def ica_plot():
    # Generate random mixed signals
    np.random.seed(0)
    n_samples = 200
    time = np.linspace(0, 8, n_samples)
    s1 = np.sin(2 * time)  # Signal 1
    s2 = np.sign(np.sin(3 * time))  # Signal 2
    s3 = np.random.randn(n_samples)  # Signal 3
    S = np.c_[s1, s2, s3]
    
    # Mixing matrix
    A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
    X = np.dot(S, A.T)  # Mixed signals
    
    # Apply ICA
    ica = FastICA(n_components=3)
    independent_components = ica.fit_transform(X)
    
<<<<<<< HEAD
    if _loss != 'focal_loss':
        _model.compile(optimizer = opt, loss=_loss, metrics=["accuracy"]) 
    else:
        _model.compile(optimizer = opt, loss=focal_loss, metrics=["accuracy"]) 
        
    #_model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=["accuracy"]) 
    #_model.compile(optimizer = opt, loss=focal_loss, metrics=["accuracy"]) 
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True)
                    
    hist = _model.fit(_X_train, _y_train, 
                      validation_data=(_X_val, _y_val), 
                      callbacks=[es], 
                      batch_size=_batch_size, 
                      epochs=_epochs, 
                      verbose=False)
    
    
    if es.stopped_epoch > 0:
        logger.info("Early stopped at epoch: " + str(es.stopped_epoch) + ' of ' + str(_epochs));

    #ev = _model.evaluate(_X_test, _y_test, verbose=False)

    res = model_predictor(_model, _X_test, _y_test)

    #_model.save(_model_name+'_tf_'+str(res['auc']), save_format='tf')
    #_model.save(_model_name+'_h5_'+str(res['auc']), save_format='h5')
    return res
=======
    # Visualize the independent components
    plt.figure(figsize=(12, 6))
>>>>>>> eea28a252afd4e7e9564e8c7e1c5d17277f6721b
    
    plt.subplot(4, 1, 1)
    plt.title("Original Signals")
    plt.plot(S)
    
    plt.subplot(4, 1, 2)
    plt.title("Mixed Signals")
    plt.plot(X)
    
    plt.subplot(4, 1, 3)
    plt.title("ICA Components")
    plt.plot(independent_components)
    
    plt.subplot(4, 1, 4)
    plt.title("Original Signals (after ICA)")
    reconstructed_signals = np.dot(independent_components, A)
    plt.plot(reconstructed_signals)
    
    plt.tight_layout()
    plt.show()
