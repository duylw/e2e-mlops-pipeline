import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status

from src.serving.model_loader import load_champion_model
from src.serving.schemas import MetadataResponse, PredictRequest, PredictResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("serving-api")

model = None
model_metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_metadata
    model = None
    model_metadata = None
    logger.info("Loading registered prediction pipeline...")
    try:
        model, model_metadata = load_champion_model()
        logger.info("Prediction pipeline loaded.")
        yield
    except Exception as exc:
        logger.critical("Failed to load prediction pipeline: %s", exc)
        yield
    finally:
        logger.info("Shutting down serving-api.")


app = FastAPI(
    title="NYC Green Taxi Trip Duration Predictor",
    description="FastAPI service for registered green taxi duration models.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction pipeline is not loaded.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.get("/metadata", response_model=MetadataResponse, status_code=status.HTTP_200_OK)
def metadata():
    if model is None or model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction pipeline is not loaded.",
        )
    return MetadataResponse(model_loaded=True, **model_metadata)


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_duration(request: PredictRequest):
    if model is None or model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serving API is not ready.",
        )

    started_at = time.time()
    try:
        df_raw = pd.DataFrame([trip.model_dump() for trip in request.trips])
        predictions = model.predict(df_raw).tolist()
        return PredictResponse(predictions=predictions, latency=time.time() - started_at, **model_metadata)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
