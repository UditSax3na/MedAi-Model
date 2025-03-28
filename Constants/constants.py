import os

# Default Paths and parameter for PredictDisease class
HOST='127.0.0.1'
PORT=8000
RELOAD=True

DEFAULT_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'\\'
DEFAULT_DATASET_PATH = 'Datasets\\'
DEFAULT_DATASET_NAME = 'disease_symptoms_binary.csv'
DEFAULT_SAVEDMODEL_PATH = DEFAULT_PARENT_DIR+'SavedModelandEncoders\\'
DEFAULT_SAVEDMODEL_NAME = 'RandomForestModel_dsb.pkl'
DEFAULT_PERCEPTRON_MODEL = 'PerceptronModel.joblib'
DEFAULT_KNN_MODEL = 'knn_model_dsb.pkl'
DEFAULT_NAIVEBAYES_MODEL = 'naive_bayes_model_dsb.pkl'
DEFAULT_ENCODER_PATH = DEFAULT_PARENT_DIR+'SavedModelandEncoders\\'
DEFAULT_ENCODER_FORPER = 'PerceptronModelEncoder.pkl'
DEFAULT_ENCODER_FORREST = 'le_disease_dsb.pkl'
DEFAULT_MODEL_NAME_1 = 'rf'
DEFAULT_MODEL_NAME_2 = 'per'
DEFAULT_MODEL_NAME_3 = 'knn'
DEFAULT_MODEL_NAME_4 = 'nb'
DEFAULT_NERMODEL = "d4data/biomedical-ner-all"
DEFAULT_AGGREGATION_STRATEGY = 'simple'
DEFAULT_LANG = 'en'
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_ESTIMATERS = 100
DEFAULT_VERBOSE = 1
DEFAULT_N_JOBS = -1
DEFAULT_N_NEIGHOURS = 5
DEFAULT_OPTIMIZER='adam'
DEFAULT_LOSS='categorical_crossentropy'
DEFAULT_METRICS=['accuracy']
DEFAULT_SYMPTOMS_PREC_DESC = DEFAULT_PARENT_DIR+DEFAULT_DATASET_PATH+'diseases_precaution_treatment.csv'