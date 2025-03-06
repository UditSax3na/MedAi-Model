import joblib
import re
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from deep_translator import GoogleTranslator
from langdetect import detect
from transformers import pipeline
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from .Constants import *

class Error(Exception):
  ''' Base Class for other exceptions '''
  pass

class DatasetNotFound(Error):
  ''' when dataset is not found on the desired path '''
  def _init_(self, message="Dataset not found on given path!"):
        self.message = message
        super()._init_(self.message)

class DatasetIsEmpty(Error):
  ''' when dataset is empty '''
  def _init_(self, message="Dataset is empty"):
        self.message = message
        super()._init_(self.message)

class ModelNotFound(Error):
  ''' when model is not found '''
  def _init_(self, message="Model Not found on given path!"):
        self.message = message
        super()._init_(self.message)

class ModelPathisEmpty(Error):
  ''' when model path is empty '''
  def _init_(self, message="Model path is empty!"):
        self.message = message
        super()._init_(self.message)

class EncoderNotFound(Error):
  ''' when encoder is not found '''
  def _init_(self, message="Encoder Not found on given path!"):
        self.message = message
        super()._init_(self.message)

class EncoderPathisEmpty(Error):
  ''' when encoder path is empty '''
  def _init_(self, message="Encoder path is empty!"):
        self.message = message
        super()._init_(self.message)

class ModeNotDefinedforXY(Error):
  ''' when the mode is not defined in _loadXY_ method '''
  def _init(self, message="Mode is not defined in __loadXY_ method"):
        self.message = message
        super()._init_(self.message)
        
class DiseaseNotFound(Error):
  ''' when disease is not found '''
  def _init_(self, message="Disease Not found in the dataset!"):
        self.message = message
        super()._init_(self.message)

class SymptomInput(BaseModel):
    text: str 
    

class DiseasePredictionModel:
  # constructor
  def __init__(self, dataset_path, model_path={'rf':DEFAULT_SAVEDMODEL_PATH+DEFAULT_SAVEDMODEL_NAME,'per':DEFAULT_SAVEDMODEL_PATH+DEFAULT_PERCEPTRON_MODEL}, model_name=[DEFAULT_MODEL_NAME_1, DEFAULT_MODEL_NAME_2], encoder_path=DEFAULT_ENCODER_PATH+DEFAULT_ENCODER_FORPER):
    self.error={}
    self.dataset = None
    self.__model_rf__ = None
    self.__model_per__ = None
    self.__X_train = None
    self.__X_test = None
    self.__y_train = None
    self.__y_test = None
    self.__history = None
    self.__model_name__ = model_name
    self.__dataset_path__ = dataset_path
    self.__model_path__ = model_path
    self.__description__ = pd.read_csv(DEFAULT_SYMPTOMS_DESC)
    self.__precautions__ = pd.read_csv(DEFAULT_SYMPTOMS_PREC)
    self.__regexpat = r"^(?:[a-zA-Z]:\\|/)?(?:[\w\-. ]+[/\\])*[\w\-. ]+$"
    self.__label_encoder__ = LabelEncoder()
    self.__encoder__ = joblib.load(encoder_path)
    self.__loadDataset__()
    self.__combined_symptoms__ = self.dataset.columns[1:]

    self.__ner_ = pipeline("ner", model=DEFAULT_NERMODEL, aggregation_strategy=DEFAULT_AGGREGATION_STRATEGY)
    if not (None in self.__model_path__):
      self.__loadModel__()
    else:
      self.__createModel__()
      self.saveModel()

  # private methods

  # method for checking whether a given path is a valid path or not using regular expression
  def __is_path__(self, path):
    return bool(re.match(self.__regexpat,path))

  # method for loading the data according to the model in x and y 
  def __loadXY__(self, mode='rf'):
    try:
      if mode=='rf':
        self.X = self.dataset.drop(columns=["disease"], )
        self.y = self.dataset["disease"]

      elif mode=='per':
        self.X = self.dataset.drop(columns=["disease"], axis=1)
        self.y = self.dataset["disease"]
        self.y = self.__encoder__.fit_transform(self.y.values.reshape(-1,1))
        self.__trainTestSplit__()

      else:
        raise ModeNotDefinedforXY

    except (ModeNotDefinedforXY) as e:
      self.error = {"location":"__loadXY__","message":e}
      print(f"Error : '{e}' from {self.error['location']}")

  # method for loading dataset
  def __loadDataset__(self):
    try:
      if self.__is_path__(self.__dataset_path__):
        self.dataset = pd.read_csv(self.__dataset_path__)
      else:
        raise DatasetNotFound

      if self.dataset.empty:
        raise DatasetIsEmpty

    except (DatasetNotFound, DatasetIsEmpty) as e:
      self.error = {"location":"__loadDataset__","message":e}
      print(f"Error : '{e}' from {self.error['location']}")

  # method for loading model
  def __loadModel__(self):
    try:
      if self.__is_path__(self.__model_path__[DEFAULT_MODEL_NAME_1]):
        self.__model_rf__ = joblib.load(self.__model_path__[DEFAULT_MODEL_NAME_1])
        self.__loadXY__(mode=DEFAULT_MODEL_NAME_1)
      else:
        raise ModelPathisEmpty
      if self.__is_path__(self.__model_path__[DEFAULT_MODEL_NAME_2]):
        self.__model_per__ = joblib.load(self.__model_path__[DEFAULT_MODEL_NAME_2])
        self.__loadXY__(mode=DEFAULT_MODEL_NAME_2)
      else:
        raise ModelPathisEmpty
    except (ModelPathisEmpty) as e:
      self.error = {"location":"_loadModel__","message":e}
      print(f"Error : '{e}' from {self.error['location']}")


  # method for train test split
  def __trainTestSplit__(self):
    self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(self.X, self.y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE)

  # method for creating a model
  def __createModel__(self):
    if 'rf' in self.__model_name__:
      self.__loadXY__(mode='rf')
      self.__model_rf__ = RandomForestClassifier(n_estimators=DEFAULT_N_ESTIMATERS, random_state=DEFAULT_RANDOM_STATE)
      self.__model_rf__.fit(self.__X_train, self.__y_train)

    if 'per' in self.__model_name__:
      self.__loadXY__(mode='per')
      self.__model_per__ = Sequential([
        Dense(DEFAULT_DENSE_LAYER_1, input_shape=(self.__X_train.shape[1],), activation=DEFAULT_FIRST_ACTIVATION_FUNC),
        Dense(10, activation=DEFAULT_FIRST_ACTIVATION_FUNC),
        Dense(self.__y_train.shape[1], activation=DEFAULT_SECOND_ACTIVATION_FUNC)
      ])
      self.__model_per__.compile(
          optimizer=DEFAULT_OPTIMIZER,
          loss=DEFAULT_LOSS,
          metrics=DEFAULT_METRICS
      )
      self.__history = self.__model_per__.fit(self.__X_train, self.__y_train, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, validation_split=DEFAULT_VALIDATION_SPLIT, verbose=DEFAULT_VERBOSE)

  # method for GetEntites from text
  def __getEntities_(self, text):
    entities = self.__ner_(text)
    for i in entities:
      print("-"*20)
      print(i)
    print("-"*20)
    print("this is e : ",[e["word"] for e in entities if e["entity_group"] == "Sign_symptom"])
    return [e["word"] for e in entities if e["entity_group"] == "Sign_symptom"]

  # public methods

  # method for predicting the disease from the model
  def predict(self, symptom):
    final = []

    # separate
    symptom_vector = np.zeros(len(self.X.columns))
    for symptom2 in symptom:
        if symptom2 in self.X.columns:
            symptom_vector[self.X.columns.get_loc(symptom2)] = 1  # Find correct index

    symptom_vector = symptom_vector.reshape(1, -1)
    print(self.__model_per__)
    prediction = self.__model_per__.predict(symptom_vector)

    predicted_index = np.argmax(prediction, axis=1)[0]
    disease_names = self.dataset['disease'].unique()

    y_train_labels = np.argmax(self.__y_train, axis=1)

    y_train_diseases = np.array([disease_names[idx] for idx in y_train_labels])
    self.__label_encoder__.fit(y_train_diseases)

    predicted_disease_pc = self.__label_encoder__.inverse_transform([predicted_index])
    final.append(predicted_disease_pc[0])

    new_symptoms_dict = {}
    for symptom1 in self.__combined_symptoms__:
        new_symptoms_dict[symptom1] = 1 if symptom1 in symptom else 0
    X_new = pd.DataFrame([new_symptoms_dict])
    predicted_disease_rf = self.__model_rf__.predict(X_new)
    final.append(predicted_disease_rf[0])

    return final

  def getPredictionFromText(self, text):
    entities = self.__ner_(text)
    STOPWORDS = {"mild", "several", "days", "of", "a", "an", "the", "his", "her", "their", "been", "for", "has", "now", "also", "feels", "in"}
    symptoms = []
    for entity in entities:
        word = entity["word"].strip()
        if entity["entity_group"] == "Sign_symptom" and word.lower() not in STOPWORDS:
            symptoms.append(word)

    SYMPTOM_KEYWORDS = {"fever", "coughing", "cold", "fatigue", "chills", "nausea", "pain", "vomiting", "headache", "congestion", "shortness of breath"}
    for word in text.lower().split():
      word_clean = word.strip(",.")
      if word_clean in SYMPTOM_KEYWORDS and word_clean not in symptoms:
          symptoms.append(word_clean)

    new_symptoms = list(set(symptoms))
    b = self.predict(new_symptoms)
    lst = []
    for i in b:
      result = {}
      result['disease'] = i
      result['Descriptions'] = self.getDescriptions(i)
      result['Precautions'] = self.getPrecautions(i) 
      lst.append(result)
    return lst
  
  # method for getting the precautions of the disease
  def getPrecautions(self, disease):
    try:
      if disease not in self.__precautions__['Disease']:
        raise DiseaseNotFound
      else:
        return self.__precautions__[self.__precautions__['Disease']==disease]['Precaution_1']['Precaution_2']  
    except (DiseaseNotFound) as e:
      self.error = {"location":"getPrecautions","message":e}
      print(f"Error : '{e}' from {self.error['location']}")
      return None
  
  # method for getting the description of the disease
  def getDescriptions(self, disease):
    try:
      if disease not in self.__description__['Disease']:
        raise DiseaseNotFound
      else:
        return self.__description__[self.__description__['Disease']==disease]   
    except (DiseaseNotFound) as e:
      self.error = {"location":"getPrecautions","message":e}
      print(f"Error : '{e}' from {self.error['location']}")
      return None
  
  # method for finding the accuracy with accuracy_score
  def accuracy(self, mode='rf'):
    if (mode=='rf'):
      prediction1 = self.__model_rf__.predict(self.__X_test)
      a = accuracy_score(self.__y_test, prediction1)
      return a
    elif (mode=='per'):
      prediction2 = self.__model_per__.predict(self.__X_test)
      b = accuracy_score(self.__y_test, prediction2)
      return b
    else:
      return None

  # method for saving the model
  def saveModel(self):
    for i in self.__model_path__:
      if i=='rf':
        joblib.dump(self.__model_rf__, DEFAULT_SAVEDMODEL_PATH + self.__model_path__[i])
      elif i=='per':
        joblib.dump(self.__model_per__, DEFAULT_SAVEDMODEL_PATH + self.__model_path__[i])
      else:
        pass

  # method for showing the model which are used
  def modelUsed(self):
    return self.__model_name__

  # method for getting the error which are stored in error dictionary
  def getError(self):
    return self.error

  # method for history
  def getHistory(self):
    return self.__history

def predict_disease(input_data: SymptomInput):
  obj = DiseasePredictionModel(
      dataset_path=DEFAULT_PARENT_DIR + DEFAULT_DATASET_PATH + DEFAULT_DATASET_NAME,
      model_path={
          'rf': DEFAULT_SAVEDMODEL_PATH + DEFAULT_SAVEDMODEL_NAME,
          'per': DEFAULT_SAVEDMODEL_PATH + DEFAULT_PERCEPTRON_MODEL
      }
  )
  result = obj.getPredictionFromText(text=input_data.text)
  
  return result