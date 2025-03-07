from fastapi import APIRouter
from .PredictDisease import SymptomInput 

router = APIRouter()

@router.get("/status")
def TestingStatus():
    return {"message": "API is live!"}

@router.post("/predict")
def Predict_disease(inputText: SymptomInput):
    from .PredictDisease import predict_disease
    result = predict_disease(SymptomInput(text=inputText.text))  
    return {"result": result}
