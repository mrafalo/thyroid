
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
    
    return sum(abs((_y_test - _y_pred) / _y_test)) / len(_y_test);

def mse_score(_y_test, _y_pred):
    
    return sum(pow(_y_test - _y_pred,2)) / len(_y_test);

def model_random_forest(_X_train, _y_train, _X_test, _y_test):
    m1 = RandomForestRegressor(n_estimators=10)
    m1 = m1.fit(_X_train, _y_train)
    mse, mape, r2, preds = model_preditor(m1, _X_test, _y_test)
    return mse, mape, r2, preds


def model_nn_custom(_layers, _sizes, _activations, _input_size):
    
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

    return m1_mse, m1_mape, m1_r2, m1_predict

    
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
    
    # Visualize the independent components
    plt.figure(figsize=(12, 6))
    
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
