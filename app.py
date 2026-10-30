from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/house_price_model.joblib")

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model not found. Run python train.py first."
    )

model = joblib.load(MODEL_PATH)

FEATURES = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "house_age",
]

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
)


class HouseInput(BaseModel):
    area_sqft: float = Field(gt=0, le=100000, examples=[1500])
    bedrooms: float = Field(ge=0, le=100, examples=[3])
    bathrooms: float = Field(ge=0, le=100, examples=[2])
    house_age: float = Field(ge=0, le=200, examples=[10])


@app.get("/")
def home():
    return {"message": "House price prediction service is running"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(house: HouseInput):
    row = pd.DataFrame(
        [house.model_dump()],
        columns=FEATURES,
    )

    prediction = float(model.predict(row)[0])

    return {"predicted_price": round(prediction, 2)}