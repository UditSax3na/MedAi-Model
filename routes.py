from fastapi import FastAPI

app = FastAPI()

@app.get("/status")
def TestingStatus():
    return {"message": "API is live!"}


@app.get("/predict")
def Predict_disease(inputText: str):
    result = predict_disease({text: inputText})
    return {"result" : result } 