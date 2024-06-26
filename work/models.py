from keras.initializers import Constant
from keras.layers import Input, Conv2D, Flatten, Activation, MaxPool2D, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add
import logging
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
import numpy as np
import utils.custom_logger as cl
from keras.utils import custom_object_scope
import keras
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152

logger = cl.get_logger()

def focal_loss(y_true, y_pred, gamma=2, alpha=2):
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt + 1e-6), axis=-1)

def model_cnn_base(_img_width, _img_height):
    # 160x160x1
    input_tensor = Input(shape=(_img_height, _img_width, 1), name="thyroid_input")
    # 160x160x8
    x = Conv2D(8, (3, 3), padding="same", activation="relu")(input_tensor)
    # 80x80x8
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 80x80x12
    x = Conv2D(12, (3, 3), padding="same", activation="relu")(x)
    # 40x40x12
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 40x40x16
    x = Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    # 20x20x16
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 20x20x24
   
    x = Conv2D(24, (3, 3), padding="same", activation="relu")(x)
    # 10x10x24
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 10x10x32
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    # 5x5x32
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    # 5x5x48
    x = Conv2D(48, (3, 3), padding="same", activation="relu")(x)
    # 5x5x48
    x = Dropout(0.5)(x)

    y_cancer = Conv2D(
        filters=1,
        kernel_size=(5, 5),
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(value=-0.9),
    )(x)

    y_cancer = Flatten()(y_cancer)
    y_cancer = Activation("sigmoid", name="out_cancer")(y_cancer)

    return Model(
        inputs=input_tensor,
        outputs=y_cancer,
    )



def model_cnn1(_img_width, _img_height):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, 1)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9)))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model

def model_cnn2(_img_width, _img_height):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, 1)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model

def model_cnn3(_img_width, _img_height):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, 1)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9)))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model

def model_cnn4(_img_width, _img_height):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, 1)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.8))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),kernel_initializer="glorot_normal",bias_initializer=Constant(value=-0.9)))
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    
    return model

def model_cnn5(_img_width, _img_height):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, 1)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (7, 7), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    
    return model

def model_cnn6(_img_width, _img_height):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, 1)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.8))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),activation='softmax'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    
    return model

def model_cnn7(_img_width, _img_height):
    model = Sequential()
    model.add(Input(shape=(_img_height, _img_width, 1)))
    model.add(Conv2D(8, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(12, (3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(16, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(1, 1)))
    model.add(Conv2D(24, (5, 5), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(32, (7, 7), padding="same", activation="relu"))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(48, (7, 7), padding="same", activation="relu"))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=1,kernel_size=(5, 5),activation='softmax'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    
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

    
    return names, models



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
    
    model = tf.keras.applications.DenseNet201(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(_img_width,_img_height, 1),
        pooling=None,
        classes=2 )
    
    return model

def model_densenet121(_img_width, _img_height):
    
    model = tf.keras.applications.DenseNet121(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(_img_width,_img_height, 1),
        pooling=None,
        classes=2 )
    
    return model

def model_VGG16(_img_width, _img_height):
    
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(_img_width,_img_height, 1),
        pooling=None,
        classes=2,
        classifier_activation="softmax")
 
    return model


def model_VGG19(_img_width, _img_height):
    
    model = tf.keras.applications.VGG19(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(_img_width,_img_height, 1),
        pooling=None,
        classes=2,
        classifier_activation="softmax")
  
 
    return model


def model_ResNet50(_img_width, _img_height):
    
    
    base_model = ResNet50(weights=None, include_top=False, input_shape=(_img_width, _img_height, 1))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)      
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

def model_ResNet101(_img_width, _img_height):
    
    base_model = ResNet101(weights=None, include_top=False, input_shape=(_img_width, _img_height, 1))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)      
    model = Model(inputs=base_model.input, outputs=predictions)
     
    return model

def model_ResNet152(_img_width, _img_height):
    
    base_model = ResNet152(weights=None, include_top=False, input_shape=(_img_width, _img_height, 1))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)      
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def find_cutoff(target, predicted):
    fpr, tpr, t = metrics.roc_curve(target, predicted)
    tnr = 1 - fpr
    g = np.sqrt(tpr*tnr)
    pos = np.argmax(g)

    return t[pos]    


def get_model_by_name(_model_name, _img_width, _img_height):
    if _model_name == 'ResNet101': return model_ResNet101(_img_width, _img_height)
    if _model_name == 'ResNet152': return model_ResNet152(_img_width, _img_height)
    if _model_name == 'ResNet50': return model_ResNet50(_img_width, _img_height)
    if _model_name == 'VGG16': return model_VGG16(_img_width, _img_height)
    if _model_name == 'VGG19': return model_VGG19(_img_width, _img_height)
    if _model_name == 'denseNet121': return model_densenet121(_img_width, _img_height)
    if _model_name == 'denseNet201': return model_densenet201(_img_width, _img_height)
    if _model_name == 'cnn1': return model_cnn1(_img_width, _img_height)    
    if _model_name == 'cnn2': return model_cnn2(_img_width, _img_height)    
    if _model_name == 'cnn3': return model_cnn3(_img_width, _img_height)    
    if _model_name == 'cnn4': return model_cnn4(_img_width, _img_height)    
    if _model_name == 'cnn5': return model_cnn5(_img_width, _img_height)    

def model_predictor(_model, _X_test, _y_test):
    
    y_base = _y_test
    _y_test = _y_test[:,1]
    
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
        
    y_predict_base = _model.predict(_X_test, verbose=0)
    m_opt_predict = y_predict_base[:,1]
    
    contains_nan = np.isnan(m_opt_predict).any()
    
    if not contains_nan:
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
    else:
        res = {
            'accuracy': 0,
            'sensitivity': 0,
            'specificity': 0,
            'precision': 0,
            'f1': 0,
            'auc': 0,
            'threshold': 0,
            'test_cases': 0,
            'test_positives': 0
            }
    
    return res


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

    with custom_object_scope({'focal_loss': focal_loss}):
        m1 = load_model(_path)
    
    return m1    

def model_fitter(_model, _X_train, _y_train, _X_val, _y_val, _X_test, _y_test, _epochs, _learning_rate, _batch_size, _optimizer, _loss, _model_name):
    
    
    if _optimizer == 'Adam':
        opt = Adam(learning_rate=_learning_rate)
    else:
        opt = SGD(learning_rate=_learning_rate)
    
    if _loss != 'focal_loss':
        _model.compile(optimizer = opt, loss=_loss, metrics=["accuracy"]) 
    else:
        _model.compile(optimizer = opt, loss=focal_loss, metrics=["accuracy"]) 
        
    #_model.compile(optimizer = opt, loss='categorical_crossentropy', metrics=["accuracy"]) 
    #_model.compile(optimizer = opt, loss=focal_loss, metrics=["accuracy"]) 
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=30, restore_best_weights=True)
                    
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
    

    
def model_compiler(_models):
    for m in _models:
        m.compile(optimizer = Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=["accuracy"]) 
                