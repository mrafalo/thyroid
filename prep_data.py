import importlib
import work
import work.data as d
import yaml    
import utils
import utils.custom_logger as cl
import cv2
import numpy as np

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    BASE_PATH = cfg['BASE_PATH']
    BASE_FILE_PATH = cfg['BASE_FILE_PATH']
    MODELING_PATH = cfg['MODELING_PATH']
    RAW_INPUT_PATH = cfg['RAW_INPUT_PATH']
    ANNOTATION_INPUT_PATH = cfg['ANNOTATION_INPUT_PATH']
    MODELING_INPUT_PATH = cfg['MODELING_INPUT_PATH']
    
    TRAIN_PATH_BASE = cfg['TRAIN_PATH_BASE']
    TEST_PATH_BASE = cfg['TEST_PATH_BASE']
    VAL_PATH_BASE = cfg['VAL_PATH_BASE']
    
    TRAIN_PATH_BW = cfg['TRAIN_PATH_BW']
    TEST_PATH_BW = cfg['TEST_PATH_BW']
    VAL_PATH_BW = cfg['VAL_PATH_BW']
    
    TRAIN_PATH_SOBEL = cfg['TRAIN_PATH_SOBEL']
    TEST_PATH_SOBEL = cfg['TEST_PATH_SOBEL']
    VAL_PATH_SOBEL = cfg['VAL_PATH_SOBEL']
    
    TRAIN_PATH_HEAT = cfg['TRAIN_PATH_HEAT']
    TEST_PATH_HEAT = cfg['TEST_PATH_HEAT']
    VAL_PATH_HEAT = cfg['VAL_PATH_HEAT']
    
    TRAIN_PATH_CANNY = cfg['TRAIN_PATH_CANNY']
    TEST_PATH_CANNY = cfg['TEST_PATH_CANNY']
    VAL_PATH_CANNY = cfg['VAL_PATH_CANNY']

    TRAIN_PATH_FELZEN = cfg['TRAIN_PATH_FELZEN']
    TEST_PATH_FELZEN = cfg['TEST_PATH_FELZEN']
    VAL_PATH_FELZEN = cfg['VAL_PATH_FELZEN']
    
    IMG_WIDTH = cfg['IMG_WIDTH']
    IMG_HEIGHT = cfg['IMG_HEIGHT']

TEST_RATIO = 0.15

logger = cl.get_logger()


logger.info('start processing...')
    
logger.info('start resizing...')

# importlib.reload(work.data)
res = d.extract_and_resize_images(ANNOTATION_INPUT_PATH, MODELING_INPUT_PATH, RAW_INPUT_PATH, IMG_WIDTH, IMG_HEIGHT)
logger.info('good! ' + str(res) + ' images resized to: (' + str(IMG_WIDTH) + ', ' + str(IMG_HEIGHT) + ') saved to: ' + MODELING_INPUT_PATH)

logger.info('start tranforming...')
res = d.mass_transformer(
    MODELING_INPUT_PATH,
    MODELING_PATH + 'all_images/',
    MODELING_PATH + 'all_images_canny/',
    MODELING_PATH + 'all_images_heat/',
    MODELING_PATH + 'all_images_sobel/',
    MODELING_PATH + 'all_images_bw/',
    MODELING_PATH + 'all_images_felzen/',
    )
logger.info('good! ' + str(res) + ' images transformed...')
