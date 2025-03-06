import uvicorn
from fastapi import FastAPI
from . import router  
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(router)  # Include router
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
