import joblib
import numpy as np
from pydantic import BaseModel
from .Constants import *
from .PredictDisease import *


class SymptomInput(BaseModel):
    text: str 

def predict_disease(input_data: SymptomInput):
    print(DEFAULT_PARENT_DIR)
    obj = DiseasePredictionModel(dataset_path=DEFAULT_PARENT_DIR + DEFAULT_DATASET_PATH + DEFAULT_DATASET_NAME, model_path={'rf':DEFAULT_SAVEDMODEL_PATH+DEFAULT_SAVEDMODEL_NAME,'per':DEFAULT_SAVEDMODEL_PATH+DEFAULT_PERCEPTRON_MODEL})
    result = obj.getPredictionFromText(text=input_data.text)
    return result 