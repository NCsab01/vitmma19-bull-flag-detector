SHAREPOINT_DATA_URL = 'https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQCDgIr7t7WJS48pAAlvANpsAQ-p-jYAWAY38H4PCihLAV4?e=6R4r2Q'

RAW_DATA_ROOT_DIR = 'data/downloaded'

TRAIN_DATA_DIR = 'data/model/train'
TEST_DATA_DIR = 'data/model/evaluation'

RESULTS_DIR = 'data/results'

INFERENCE_DIR = 'data/downloaded/datas/inference'

X_TRAIN_NAME = 'X_train.npy'
Y_TRAIN_NAME = 'Y_train.npy'
L_TRAIN_NAME = 'L_train.npy'

X_VAL_NAME = 'X_val.npy'
Y_VAL_NAME = 'Y_val.npy'
L_VAL_NAME = 'L_val.npy'

X_TEST_NAME = 'X_test.npy'
Y_TEST_NAME = 'Y_test.npy'
L_TEST_NAME = 'L_test.npy'

MODEL_PARAMETERS_DIR = 'data/model/parameters'
MODEL_PARAMETERS_PATH = 'flag_classifier.pth'

MAX_SEQUENCE_LENGTH = 128
MIN_SEQUENCE_LENGTH = 32

INPUT_SIZE = 4
NUM_CLASSES = 6

HIDDEN_SIZE = 64  
NUM_LAYERS = 2    

DROPOUT = 0.5     

EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

LABEL_MAP = {
    "Bullish Normal": 0, "Bullish Wedge": 1, "Bullish Pennant": 2,
    "Bearish Normal": 3, "Bearish Wedge": 4, "Bearish Pennant": 5
}