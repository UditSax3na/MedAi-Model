from fastapi import APIRouter
from MedAiModelmain.PredictDisease import SymptomInput 

router = APIRouter()

@router.get("/status")
def TestingStatus():
    return {"message": "API is live!"}

@router.post("/predict")
def Predict_disease(inputText: SymptomInput):
    from MedAiModelmain.PredictDisease import predict_disease
    result = predict_disease(SymptomInput(text=inputText.text))  
    return {"result": result}
