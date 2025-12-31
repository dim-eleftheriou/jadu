import logging
import numpy as np

import ollama

from typing import Optional
from pydantic import BaseModel

from pydantic.json_schema import JsonSchemaValue
from pydantic import BaseModel, ValidationError

from fastapi import FastAPI, Request

import time
import yaml
import pickle
import joblib

#-----------------------------------
#      Risk Estimator Setup
#-----------------------------------

class RiskEstimator:
    def __init__(self, embedding_model_name, risk_model_name, encoder_name, embedding_model_options):

        self.embedding_model_name = embedding_model_name
        self.risk_model_name = risk_model_name
        self.encoder_name = encoder_name
        self.embedding_model_options = embedding_model_options
        self.risk_model = joblib.load(self.risk_model_name)
        self.encoder = joblib.load(self.encoder_name)

        self.client = ollama.Client(host=self.embedding_model_options.get("base_url", "http://localhost:11434"))

    def process(self, job: str) -> np.ndarray:

        # Create embedding features
        X_embeddings = self.client.embeddings(
            model=self.embedding_model_name,
            prompt=job['prompt'],
            options=self.embedding_model_options
        ).embedding
        X_embeddings = np.array(X_embeddings).reshape(1, -1)

        # Create modelId features
        modelId = np.array([job['modelId']]).reshape(1, -1)
        X_modelId = self.encoder.transform(modelId)
        X_modelId = X_modelId.toarray()

        features = np.concatenate([X_embeddings, X_modelId], axis=1)
        return features

    def estimate_risk(self, job):
        features = self.process(job)
        risk = self.risk_model.predict_proba(features)[:, 1][0]
        return risk.item()

#--------------------------------------
#      Risk Estimator Initialization 
#--------------------------------------
with open("risk_estimator_config.yaml", "r") as f:
    config = yaml.safe_load(f)

estimator = RiskEstimator(**config)

#-----------------------------------
#          Logger Setup 
#-----------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("risk_calculation.log"),   # Save logs to file
        logging.StreamHandler()           # Still print to console
    ]
)
logger = logging.getLogger(__name__)

#-----------------------------------
#        FastAPI App Setup 
#-----------------------------------

app = FastAPI()

# Middleware for logging requests & responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    if request.method in ("POST"):
        body = await request.body()
        logger.info(f"Request body: {body.decode('utf-8')}")
    
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

#-----------------------------------
#         Data Models 
#-----------------------------------

# Data model of input data
class InputData(BaseModel):
    jobId: Optional[str] = None
    createdAt: Optional[dict] = None
    status: Optional[str] = None
    modelTitle: Optional[str] = None
    modelId: str
    modelMetadata: Optional[dict] = None
    prompt: str
    images: Optional[list] = None
    outputImage: Optional[str] = None
    qaScore: Optional[int] = None
    qaTransformedScore: Optional[int] = None
    qaReasoning: Optional[str] = None
    qaActionableFeedback: Optional[str] = None

# Data model of output data
class OutputData(BaseModel):
    risk: float
    model_used: str
    execution_time: float
    status: str
    error: str

#-----------------------------------
#          API Endpoints 
#-----------------------------------

# POST requests by
# curl -X POST "http://localhost:8080/calculate_risk" -H "Content-Type: application/json" -d '{...}'
@app.post("/calculate_risk", response_model=OutputData)
async def calculate_risk(input_data: InputData):
    start = time.time()
    try:
        risk = estimator.estimate_risk(input_data.dict())
        status = "SUCCESS"
        error = ""
    except Exception as e:
        risk = -1
        status = "FAILED"
        error = str(e)
    end = time.time()
    logger.info(f"RISK CALCULATION STATUS: {status}")
    if status != "SUCCESS":
        logger.error(f"RISK CALCULATION ERROR: {error}")
    logger.info(f"RISK CALCULATION PROCESSING TIME: {end-start} seconds")

    result = {
        "risk": risk,
        "model_used": estimator.risk_model_name,
        "execution_time": end-start,
        "status": status,
        "error": error
    }
    return result

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
