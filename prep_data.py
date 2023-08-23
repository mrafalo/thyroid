import importlib
import work
import work.data as d
import yaml    
import logging

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    BASE_PATH = cfg['BASE_PATH']
    IMG_WIDTH = cfg['IMG_WIDTH']
    IMG_HEIGHT = cfg['IMG_HEIGHT']


logger = logging.getLogger('data_prep')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s(%(name)s) %(levelname)s: %(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
if (logger.hasHandlers()):
    logger.handlers.clear()
  
logger.addHandler(ch)

logger.info('start processing...')
    
TEST_RATIO = 0.15

BASE_FILE_PATH = BASE_PATH + 'baza4.csv'
INPUT_PATH = BASE_PATH + 'source/'
OUTPUT_PATH = BASE_PATH + 'resized/'
TRAIN_PATH = BASE_PATH + 'modeling/all_images/train/'
TEST_PATH = BASE_PATH + 'modeling/all_images/test/'

TRAIN_PATH_BW = BASE_PATH + 'modeling/all_images_bw/train/'
TEST_PATH_BW = BASE_PATH + 'modeling/all_images_bw/test/'

TRAIN_PATH_SOBEL = BASE_PATH + 'modeling/all_images_sobel/train/'
TEST_PATH_SOBEL = BASE_PATH + 'modeling/all_images_sobel/test/'

TRAIN_PATH_HEAT = BASE_PATH + 'modeling/all_images_heat/train/'
TEST_PATH_HEAT = BASE_PATH + 'modeling/all_images_heat/test/'

TRAIN_PATH_CANNY = BASE_PATH + 'modeling/all_images_canny/train/'
TEST_PATH_CANNY = BASE_PATH + 'modeling/all_images_canny/test/'

TRAIN_PATH_FELZEN = BASE_PATH + 'modeling/all_images_felzen/train/'
TEST_PATH_FELZEN = BASE_PATH + 'modeling/all_images_felzen/test/'

 
# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(utils.image_manipulator)
logger.info('start resizing...')
res = d.resize_images(INPUT_PATH, OUTPUT_PATH, IMG_WIDTH, IMG_HEIGHT)

logger.info('good! ' + str(res) + ' images resized to: (' + str(IMG_WIDTH) + ', ' + str(IMG_HEIGHT) + ') saved to: ' + OUTPUT_PATH)

logger.info('start tranforming...')

res = d.mass_transformer(
    OUTPUT_PATH,
    BASE_PATH + 'modeling/all_images/',
    BASE_PATH + 'modeling/all_images_canny/',
    BASE_PATH + 'modeling/all_images_heat/',
    BASE_PATH + 'modeling/all_images_sobel/',
    BASE_PATH + 'modeling/all_images_bw/',
    BASE_PATH + 'modeling/all_images_felzen/',
    )

logger.info('good! ' + str(res) + ' images transformed...')
logger.info('start splitting...')

d.split_files(BASE_PATH + 'modeling/all_images/', TRAIN_PATH, TEST_PATH, TEST_RATIO)
d.split_files(BASE_PATH + 'modeling/all_images_bw/', TRAIN_PATH_BW, TEST_PATH_BW, TEST_RATIO)
d.split_files(BASE_PATH + 'modeling/all_images_sobel/', TRAIN_PATH_SOBEL, TEST_PATH_SOBEL, TEST_RATIO)
d.split_files(BASE_PATH + 'modeling/all_images_heat/', TRAIN_PATH_HEAT, TEST_PATH_HEAT, TEST_RATIO)
d.split_files(BASE_PATH + 'modeling/all_images_canny/', TRAIN_PATH_CANNY, TEST_PATH_CANNY, TEST_RATIO)
d.split_files(BASE_PATH + 'modeling/all_images_felzen/', TRAIN_PATH_FELZEN, TEST_PATH_FELZEN, TEST_RATIO)

logger.info('good! images split at ratio: ' + str(TEST_RATIO) + ' ALL OK!')
