import pandas as pd

from typing import List, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from challenge.model import DelayModel, FeaturesNotFound, ModelNotFound

app = FastAPI()
model = DelayModel()


class Flights(BaseModel):
    """Flights data class for data POST"""
    flights: List[Dict]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(request: Request, data: Flights) -> dict:
    # Reading POST data and preparing it for prediction
    df = pd.DataFrame(data.flights)
    df_processed = model.preprocess(df)

    # Predict
    predictions = model.predict(df_processed)
    response = {'predict': predictions}

    return response


@app.exception_handler(FeaturesNotFound)
async def features_not_found(request: Request, fnf: FeaturesNotFound):
    return JSONResponse(
        status_code=400,
        content={
            "message": "Predict data was not found on model features.",
            "features": fnf.features,
        },
    )


@app.exception_handler(ModelNotFound)
async def model_not_found(request: Request, mnf: ModelNotFound):
    return JSONResponse(
        status_code=400,
        content={"message": mnf.message},
    )


@app.exception_handler(Exception)
async def general_exception(request: Request, e: Exception):
    return JSONResponse(
        status_code=400,
        content={
            "message": "There was a general error on the API",
            "details": str(e),
        },
    )