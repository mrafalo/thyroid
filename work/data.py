import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import os
from imgaug import augmenters
from tensorflow.keras.utils import to_categorical
import random
import shutil
import utils.image_manipulator as im
import fnmatch
import np_utils
from sklearn.model_selection import train_test_split

def augment(_dataset):
    # data augmentation
    seq = augmenters.Sequential(
        [
            augmenters.Fliplr(0.5),
            augmenters.Flipud(0.5),
            augmenters.Affine(rotate=(-15, 15)),
            augmenters.Affine(shear=(-15, 15)),
            augmenters.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            augmenters.Affine(scale=(0.9, 1.1)),
        ]
    )
    return seq.augment_images(_dataset)

def extract_image(_image, _image_path):
    IMG_BORDER = 10
    raw_image = cv2.imread(_image_path)
    
    height, width = _image.shape[:2]
    hsv = _image#cv2.cvtColor(_image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 0, 200])
    upper_red = np.array([120, 120, 255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    # el = np.array([contour[0]])
    # contour = np.concatenate((contour, el))

    #cv2.drawContours(_image, contour, -1, (0,0, 255), 2)  
    cv2.polylines(_image, [contour], isClosed=True, color=(0, 0, 255), thickness=2)
    
    hsv = _image
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]

    bw_mask = np.zeros_like(raw_image)
    cv2.fillPoly(bw_mask, pts=[contour], color=(255, 255, 255))
    raw_image = cv2.bitwise_and(raw_image, bw_mask)

    x, y, w, h = cv2.boundingRect(contour)
    y1 = y - IMG_BORDER if y - IMG_BORDER > 0 else 0
    y2 = y + h + IMG_BORDER if y + IMG_BORDER + h < height else height
    x1 = x - IMG_BORDER if x - IMG_BORDER > 0 else 0
    x2 = x + w + IMG_BORDER if x + IMG_BORDER + w < width else width
    
    return raw_image[y1:y2, x1:x2]


def resize_with_aspect_ratio(_image, width=None, height=None):
    (h, w) = _image.shape[:2]

    if width is None and height is None:
        return _image

    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image
    resized = cv2.resize(_image, dim, interpolation=cv2.INTER_AREA)

    # Check if we need to add padding
    delta_w = width - resized.shape[1]
    delta_h = height - resized.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    top = 0 if top < 0 else top
    bottom = 0 if bottom < 0 else bottom
    left = 0 if left < 0 else left
    right = 0 if right < 0 else right
    
    color = [0, 0, 0]  # Black padding
    resized_with_padding = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    resized_with_padding = cv2.resize(resized_with_padding, (width, height), interpolation=cv2.INTER_AREA)
    return resized_with_padding
    
def extract_and_resize_images(_input_path, _output_path, _raw_path, _width, _height):
    res = 0
    for f in os.listdir(_input_path):
        filename = os.fsdecode(f)
        if filename.endswith(".png"): 
            print('processing', f, 'to', f.replace("out_shape_", ""))
            image = cv2.imread(_input_path+f)
            image = extract_image(image, _raw_path + filename.replace("out_shape_", ""))
            image = im.blur_manual(image, 2, 1)
            # resized = cv2.resize(image, (_width, _heigt), interpolation = cv2.INTER_AREA)
            resized = resize_with_aspect_ratio(image, _width, _height)
            gray = resized #cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(_output_path+'resized_' + f, gray)
            #print('processing', f, 'to', f.replace("out_shape_", ""))
            res = res + 1
    return res

def resize_images(_input_path, _output_path, _width, _height):
    
    res = 0
    for f in os.listdir(_input_path):
        filename = os.fsdecode(f)
        if filename.endswith(".png"): 
            image = cv2.imread(_input_path+f)
            image = im.blur_manual(image, 2, 1)
            resized = cv2.resize(image, (_width, _height), interpolation = cv2.INTER_AREA)
            gray = resized #cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(_output_path+'resized_' + f, gray)
            res = res + 1
    return res

def split_files(_source_path, _train_path, _val_path, _test_path, _val_ratio, _test_ratio):

    
    paths = [_train_path, _test_path, _val_path]
    for p in paths:
        isExist = os.path.exists(p)
        if not isExist:
            os.makedirs(p)
    
    for f in os.listdir(_train_path):
       file_path = os.path.join(_train_path, f)
       if os.path.isfile(file_path):
         os.remove(file_path)

    for f in os.listdir(_test_path):
       file_path = os.path.join(_test_path, f)
       if os.path.isfile(file_path):
         os.remove(file_path)
         
    for f in os.listdir(_val_path):
       file_path = os.path.join(_val_path, f)
       if os.path.isfile(file_path):
         os.remove(file_path)
         
    number_of_test_files = int(np.round(_test_ratio * len(fnmatch.filter(os.listdir(_source_path), '*.png'))))
    number_of_val_files = int(np.round(_val_ratio * len(fnmatch.filter(os.listdir(_source_path), '*.png'))))
   
    test_list = []
    for i in range(number_of_test_files):
        random_file = random.choice(os.listdir(_source_path))
        filename = os.fsdecode(random_file)
        if filename.endswith(".png"): 
            shutil.copy(_source_path + random_file, _test_path + random_file)
            test_list.append(filename)
                   
    val_list = []
    for i in range(number_of_val_files):
        random_file = random.choice(os.listdir(_source_path))
        filename = os.fsdecode(random_file)
        if filename.endswith(".png"): 
            shutil.copy(_source_path + random_file, _val_path + random_file)
            val_list.append(filename)
       
    for f in os.listdir(_source_path):
        filename = os.fsdecode(f)
        if filename.endswith(".png"): 
            if (not filename in val_list) and (not filename in test_list):
                shutil.copy(_source_path + f, _train_path + f)
        

def mass_transformer(_source_path, _dest_path_base, _dest_path_canny, _dest_path_heat,  _dest_path_sobel, _dest_path_bw, _dest_path_felzen):
    res = 0
    paths = [_dest_path_base, _dest_path_canny, _dest_path_heat,  _dest_path_sobel, _dest_path_bw, _dest_path_felzen]
    for p in paths:
        isExist = os.path.exists(p)
        if not isExist:
            os.makedirs(p)
   
    for f in os.listdir(_source_path):
        filename = os.fsdecode(f)
        if filename.endswith(".png"):
            img = cv2.imread(_source_path + f)
            canny = im.edges(img, 30, 105)
            heat = im.heatmap(img)
            sobel = im.sobel(img)
            bw = im.bw_mask(img)
            felzen = im.felzenszwalb(img, 110)
            
            cv2.imwrite(_dest_path_base + 'base_'+  f, img)
            cv2.imwrite(_dest_path_canny + 'canny_'+  f, canny)
            cv2.imwrite(_dest_path_heat + 'heat_'+  f, heat)
            cv2.imwrite(_dest_path_sobel + 'sobel_'+  f, sobel)
            cv2.imwrite(_dest_path_bw + 'bw_'+  f, bw)
            cv2.imwrite(_dest_path_felzen + 'felzen_'+  f, felzen)
            
            res = res + 1
            
    return res

            
                    
def malignancy_splitter(_data_file, _source_path, _dest_path_benign, _dest_path_malignant,  _transformation):
    df = pd.read_csv(_data_file, sep=';')
    
    
    for f in os.listdir(_source_path):
        filename = os.fsdecode(f)
        if filename.endswith(".png"): 
            
            f_slit = f.split('_')
            id_coi = f_slit[5]
            
            rak = df.loc[(df.id_coi==id_coi) ,'rak'].iloc[0]
            
            if _transformation == 'canny':
                img = cv2.imread(_source_path + f)
                edge = im.edges(img, 30, 105)
                if rak==1:
                    cv2.imwrite(_dest_path_malignant+'canny_'+f, edge)
                else:
                    cv2.imwrite(_dest_path_benign+'canny_'+f, edge)
            
            elif _transformation == 'heat':
                img = cv2.imread(_source_path + f)
                edge = im.heatmap(img)
                if rak==1:
                    cv2.imwrite(_dest_path_malignant+'heat_'+f, edge)
                else:
                    cv2.imwrite(_dest_path_benign+'heat_'+f, edge)
            
            elif _transformation == 'sobel':
                img = cv2.imread(_source_path + f)
                edge = im.sobel(img)
                if rak==1:
                    cv2.imwrite(_dest_path_malignant+'sobel_'+f, edge)
                else:
                    cv2.imwrite(_dest_path_benign+'sobel_'+f, edge)
            
            elif _transformation == 'bw':
                img = cv2.imread(_source_path + f)
                edge = im.bw_mask(img)
                if rak==1:
                    cv2.imwrite(_dest_path_malignant+'bw_'+f, edge)
                else:
                    cv2.imwrite(_dest_path_benign+'bw_'+f, edge)

            elif _transformation == 'felzen':
                img = cv2.imread(_source_path + f)
                edge = im.felzenszwalb(img, 110)
                if rak==1:
                    cv2.imwrite(_dest_path_malignant+'felzen_'+f, edge)
                else:
                    cv2.imwrite(_dest_path_benign+'felzen_'+f, edge)
                    
            else:
                if rak==1:
                    shutil.copy(_source_path + f, _dest_path_malignant)
                else: 
                    shutil.copy(_source_path + f, _dest_path_benign)

def label_cancer (row):
   if row['HP_PTC'] == 1 :
      return 'PTC'
   if row['HP_FTC'] == 1 :
      return 'FTC'
   if row['HP_Hurthlea'] == 1:
      return 'HURTHLEA'
   if row['HP_MTC']  == 1:
      return 'MTC'
   if row['HP_DOBRZE_ZROZNICOWANE'] == 1:
      return 'DOBRZE_ZROZNICOWANY'
   if row['HP_ANA'] == 1:
      return 'ANAPLASTYCZNY'
   if row['HP_PLASKO'] == 1:
      return 'PLASKONABLONKOWY'
   else:
    return 'BENIGN'

def load_data_file(_data_file):
    df = pd.read_csv(_data_file, sep=';')
    df['label_cancer'] = df.apply (lambda row: label_cancer(row), axis=1)
    
    
    zmienne = ['echo_nieznacznie_hipo', 'echo_gleboko_hipo', 'echo_hiperechogeniczna',
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
    'wezly_chlonne_patologiczne',
    'lokalizacja_prawy_plat', 'lokalizacja_lewy_plat', 'lokalizacja_ciesn',
    'HP_PTC', 'HP_FTC', 'HP_Hurthlea', 'HP_MTC', 'HP_DOBRZE_ZROZNICOWANE', 'HP_ANA', 'HP_PLASKO',	
    'HP_RUCZOLAK', 'HP_GUZEK_ROZROSTOWY', 'HP_ZAPALENIE', 'HP_NIEOKRESLONE', 'HP_NIFTP', 'HP_WDUMP','HP_FTUMP',
    'rak']
    
    
    df = df.fillna(-1)
    df[zmienne] = df[zmienne].astype(int)

    df.loc[df.BACC_2==1, 'BACC_Bethesda']='kat2'
    df.loc[df.BACC_3==1, 'BACC_Bethesda']='kat3'
    df.loc[df.BACC_4==1, 'BACC_Bethesda']='kat4'
    df.loc[df.BACC_5==1, 'BACC_Bethesda']='kat5'
    df.loc[df.BACC_6==1, 'BACC_Bethesda']='kat6'

    df.loc[df.tirads_2==1, 'tirads']='2'
    df.loc[df.tirads_3==1, 'tirads']='3'
    df.loc[df.tirads_4==1, 'tirads']='4'
    df.loc[df.tirads_5==1, 'tirads']='5'
    
    df['max_dim'] = df[['szerokosc', 'grubosc', 'dlugosc']].max(axis=1)

    return df

    
def split_data_4cancer(_data_file, _base_path, _augument, _val_ratio, _test_ratio, _img_size, _seed=123):
    
    df = load_data_file(_data_file)

    X = []
    y = []
    
    for f in os.listdir(_base_path):
        f_slit = f.split('_')
    
        id_coi = f_slit[4]
     
        if len(df.loc[(df.id_coi==id_coi) ,'rak']) > 0:
            rak = df.loc[(df.id_coi==id_coi) ,'rak'].iloc[0]
            y.append(rak)
            im = cv2.imread(_base_path + f, cv2.IMREAD_GRAYSCALE)
            resized = resize_with_aspect_ratio(im, _img_size, _img_size)
            print('base:', im.shape)
            print('resized:', resized.shape)
            
            X.append(np.array(resized))
            
            #print('id_coi:', id_coi, 'file:', f, 'rak:', rak)
        else:
            raise ValueError("Patient id_coi:", id_coi, 'not found!')
    

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_ratio, random_state=_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=_val_ratio, random_state=_seed)
    
    if _augument > 0:
        X_train_tmp = X_train
        y_train_tmp = y_train
    
        for i in range(0, _augument):
            X_train_augumented = augment(X_train)
            X_train_tmp = X_train_tmp + X_train_augumented
            y_train_tmp = y_train_tmp + y_train
            
        X_train = X_train_tmp
        y_train = y_train_tmp
    
    train_size = len(X_train)
    test_size = len(X_test)
    val_size = len(X_val)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    im_width = X_train[0].shape[0]
    im_height = X_train[0].shape[1]
    X_train = X_train.reshape(train_size,im_width,im_height,1)
    X_val = X_val.reshape(val_size,im_width,im_height,1)
    X_test = X_test.reshape(test_size,im_width,im_height,1)
    
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_val /= 255
    X_test /= 255

    nb_classes = 2
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)
    y_test = to_categorical(y_test, nb_classes)


    return X_train, y_train, X_val, y_val, X_test, y_test       
                    
def split_data_4feature(_data_file, _base_path, _augument, _val_ratio, _test_ratio, _feature, _seed=123):
    
    df = load_data_file(_data_file)

    X = []
    y = []
    
    for f in os.listdir(_base_path):
        f_slit = f.split('_')
    
        id_coi = f_slit[4]
     
        if len(df.loc[(df.id_coi==id_coi) ,_feature]) >0:
            feature_val = df.loc[(df.id_coi==id_coi) ,_feature].iloc[0]
            y.append(feature_val)
            X.append(np.array(cv2.imread(_base_path + f, cv2.IMREAD_GRAYSCALE)))
        else:
            raise ValueError("Patient id_coi:", id_coi, 'not found!')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_test_ratio, random_state=_seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=_val_ratio, random_state=_seed)
    
    if _augument > 0:
        X_train_tmp = X_train
        y_train_tmp = y_train
    
        for i in range(0, _augument):
            X_train_augumented = augment(X_train)
            X_train_tmp = X_train_tmp + X_train_augumented
            y_train_tmp = y_train_tmp + y_train
            
        X_train = X_train_tmp
        y_train = y_train_tmp
    
    train_size = len(X_train)
    test_size = len(X_test)
    val_size = len(X_val)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    im_width = X_train[0].shape[0]
    im_height = X_train[0].shape[1]
    X_train = X_train.reshape(train_size,im_width,im_height,1)
    X_val = X_val.reshape(val_size,im_width,im_height,1)
    X_test = X_test.reshape(test_size,im_width,im_height,1)
    
    
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')
    
    X_train /= 255
    X_val /= 255
    X_test /= 255
    
    nb_classes = 2
    y_train = to_categorical(y_train, nb_classes)
    y_val = to_categorical(y_val, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test        
                    
                  
def img_to_predict(_file_path):

    X_val = np.array(cv2.imread(_file_path , cv2.IMREAD_GRAYSCALE))
    
    
    im_width = X_val.shape[0]
    im_height = X_val.shape[1]
    
    X_val = X_val.reshape(1,im_width,im_height,1)
    X_val = X_val.astype('float32')
    X_val /= 255

    return X_val        
                    




