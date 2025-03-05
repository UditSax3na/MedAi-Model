import uvicorn
from fastapi import FastAPI
from code.routes import router  # Import router
from Constant import HOST, PORT, RELOAD

app = FastAPI()
app.include_router(router) 

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)