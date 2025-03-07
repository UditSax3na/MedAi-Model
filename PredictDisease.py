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
from scipy.special import softmax
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
  def __init__(self, dataset_path, model_path={'rf':DEFAULT_SAVEDMODEL_PATH+DEFAULT_SAVEDMODEL_NAME,'per':DEFAULT_SAVEDMODEL_PATH+DEFAULT_PERCEPTRON_MODEL,'knn':DEFAULT_SAVEDMODEL_PATH+DEFAULT_KNN_MODEL,'nb':DEFAULT_SAVEDMODEL_PATH+DEFAULT_NAIVEBAYES_MODEL}, model_name=[DEFAULT_MODEL_NAME_1, DEFAULT_MODEL_NAME_2], encoder_path=DEFAULT_ENCODER_PATH+DEFAULT_ENCODER_FORPER):
    self.error={}
    self.weights = {
      'rf': 0.4,  # Random Forest
      'knn': 0.3,  # k-NN
      'nb': 0.2,  # Naive Bayes
      'mlp': 0.1   # Perceptron
    }
    self.dataset = None
    self.__model_rf__ = None
    self.__model_knn__ = None
    self.__model_nb__ = None
    self.__X_train = None
    self.__X_test = None
    self.__y_train = None
    self.__y_test = None
    self.__history = None
    self.__model_name__ = model_name
    self.__dataset_path__ = dataset_path
    self.__model_path__ = model_path
    self.__dpt__ = pd.read_csv(DEFAULT_SYMPTOMS_PREC_DESC)
    self.__regexpat = r"^(?:[a-zA-Z]:\\|/)?(?:[\w\-. ]+[/\\])*[\w\-. ]+$"
    self.__label_encoder__ = LabelEncoder()
    self.__encoder_forrest__ = joblib.load(DEFAULT_ENCODER_PATH+DEFAULT_ENCODER_FORREST)
    self.__encoder__ = joblib.load(encoder_path)
    self.__loadDataset__()
    self.__combined_symptoms__ = self.dataset.columns[1:]

    self.__ner_ = pipeline("ner", model=DEFAULT_NERMODEL, aggregation_strategy=DEFAULT_AGGREGATION_STRATEGY)
    self.__loadModel__()

  # private methods

  # method for checking whether a given path is a valid path or not using regular expression
  def __is_path__(self, path):
    return bool(re.match(self.__regexpat,path))

  # method for loading the data according to the model in x and y 
  def __loadXY__(self):
    self.X = self.dataset.drop(columns=["disease"], axis=1)
    self.y = self.__encoder_forrest__.fit_transform(self.dataset["disease"])
    self.__label_encoder__.fit(self.dataset["disease"])  # Ensure LabelEncoder is trained
    self.__trainTestSplit__(X=self.X, y=self.y, mode='rf')

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
        self.__loadXY__()
      else:
        raise ModelPathisEmpty
      if self.__is_path__(self.__model_path__[DEFAULT_MODEL_NAME_3]):
        self.__model_knn__ = joblib.load(self.__model_path__[DEFAULT_MODEL_NAME_3])
        self.__loadXY__()
      else:
        raise ModelPathisEmpty
      if self.__is_path__(self.__model_path__[DEFAULT_MODEL_NAME_4]):
        self.__model_nb__ = joblib.load(self.__model_path__[DEFAULT_MODEL_NAME_4])
        self.__loadXY__()
      else:
        raise ModelPathisEmpty
    except (ModelPathisEmpty) as e:
      self.error = {"location":"_loadModel__","message":e}
      print(f"Error : '{e}' from {self.error['location']}")


  # method for train test split
  def __trainTestSplit__(self, X, y, mode='per'):
    if mode=='per':
      pass
    else:
      self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE)

  # deprecated method
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

    # Prepare the symptom vector for Random Forest, k-NN, and Naive Bayes
    new_symptoms_dict = {}
    for symptom1 in self.__combined_symptoms__:
        new_symptoms_dict[symptom1] = 1 if symptom1 in symptom else 0
    X_new = pd.DataFrame([new_symptoms_dict])

    # Get predictions from Random Forest, k-NN, and Naive Bayes
    predicted_disease_rf = self.__model_rf__.predict(X_new)[0]
    predicted_disease_knn = self.__model_knn__.predict(X_new)[0]
    predicted_disease_nb = self.__model_nb__.predict(X_new)[0]

    final.append(predicted_disease_rf)
    final.append(predicted_disease_knn)
    final.append(predicted_disease_nb)

    # Get predicted probabilities from each model
    y_proba_rf = self.__model_rf__.predict_proba(X_new)
    y_proba_knn = self.__model_knn__.predict_proba(X_new)
    y_proba_nb = self.__model_nb__.predict_proba(X_new)
    
    # Combine probabilities using weighted averaging
    weights = {
      'rf': 0.5,  # Random Forest
      'knn': 0.3,  # k-NN
      'nb': 0.2   # Naive Bayes
    }

    y_proba_ensemble = (
      weights['rf'] * y_proba_rf +
      weights['knn'] * y_proba_knn +
      weights['nb'] * y_proba_nb
    )

    # Get the final predicted class
    y_pred_ensemble = np.argmax(y_proba_ensemble, axis=1)[0]
    predicted_disease_ensemble = self.__label_encoder__.inverse_transform([y_pred_ensemble])[0]

    # Return the final prediction
    return predicted_disease_ensemble

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
    output = self.get_disease_info(b, self.__dpt__)
    return output
  
  def get_disease_info(self,disease_name, df):
    disease_data = df[df['Disease'].str.lower() == disease_name.lower()]
    
    if disease_data.empty:
      return {"Error": "Disease not found in the dataset."}
    
    precautions = disease_data[disease_data['Type'] == 'Precaution']['Precaution/Treatment'].tolist()
    treatments = disease_data[disease_data['Type'] == 'Treatment']['Precaution/Treatment'].tolist()
    
    return {
      "Disease": disease_name,
      "Precautions": precautions if precautions else ["No precautions listed"],
      "Treatments": treatments if treatments else ["No treatments listed"]
    }

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
      model_path={'rf':DEFAULT_SAVEDMODEL_PATH+DEFAULT_SAVEDMODEL_NAME,'per':DEFAULT_SAVEDMODEL_PATH+DEFAULT_PERCEPTRON_MODEL,'knn':DEFAULT_SAVEDMODEL_PATH+DEFAULT_KNN_MODEL,'nb':DEFAULT_SAVEDMODEL_PATH+DEFAULT_NAIVEBAYES_MODEL}
  )
  result = obj.getPredictionFromText(text=input_data.text)
  
  return result