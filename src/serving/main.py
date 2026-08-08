import os
import time
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager

from src.serving.schemas import PredictRequest, PredictResponse
from src.serving.model_loader import load_champion_model_and_preprocessor
from src.preprocessing import clean_taxi_data_inference, engineer_base_features, transform_advanced_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("serving-api")

# Global containers for loaded artifacts
model = None
preprocessor = None
version = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler to load the model and preprocessor once
    at startup, and clean up at shutdown.
    """
    global model, preprocessor, version
    logger.info("Initializing serving-api, loading model and preprocessor...")
    try:
        model, preprocessor, version = load_champion_model_and_preprocessor()
        logger.info("Model and preprocessor loaded successfully. Ready to serve.")
        yield
    except Exception as e:
        logger.critical(f"FATAL: Failed to load champion model/preprocessor from MLflow: {e}")
        # Allow server to start but endpoints will return 503 Service Unavailable
        yield
    finally:
        logger.info("Shutting down serving-api.")

app = FastAPI(
    title="NYC Green Taxi Trip Duration Predictor",
    description="Production API serving green taxi duration predictions.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Health check endpoint to verify MLflow connection status.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or preprocessor is not loaded. Check MLflow server connection."
        )
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict_duration(request: PredictRequest):
    """
    Accepts raw inference payloads, processes advanced features, and returns model predictions.
    """
    start_time = time.time()
    
    # 1. Check if model and preprocessor are loaded
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serving API is not ready. Model or preprocessor not loaded."
        )
        
    num_trips = len(request.trips)
    logger.info(f"Received predict request containing {num_trips} trip(s).")
    
    try:
        # 2. Convert Pydantic list of inputs to pandas DataFrame
        trips_dict = [trip.model_dump() for trip in request.trips]
        df_raw = pd.DataFrame(trips_dict)
        
        # 3. Clean raw inference input
        df_clean = clean_taxi_data_inference(
            df_raw, trip_type_mode_fallback=preprocessor.get('trip_type_mode', 1)
        )
        
        # 4. Extract base features using coordinates lookup
        df_lookup = preprocessor.get('df_lookup')
        if df_lookup is None:
            raise RuntimeError("df_lookup coordinate table is missing in preprocessor artifacts.")
            
        df_base = engineer_base_features(df_clean, df_lookup)
        if df_base.empty:
            logger.warning("All records were filtered out during coordinate lookup merge.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="All records were filtered out during coordinate lookup merge. "
                       "Check if PULocationID and DOLocationID are valid NYC TLC zone IDs."
            )
            
        # 5. Extract advanced features mapping speed profile and categorical list
        df_final = transform_advanced_features(df_base, preprocessor)
        
        # 6. Extract features matrix aligned with training columns
        X_infer = df_final.drop(columns=['duration'], errors='ignore')
        X_infer = X_infer.reindex(columns=preprocessor['train_columns'], fill_value=0)
        
        # 7. Predict duration in minutes
        preds = model.predict(X_infer)
        predictions_list = preds.tolist()
        
        duration = time.time() - start_time
        logger.info(f"Successfully processed {num_trips} prediction(s) in {duration:.4f} seconds.")
        logger.info(f"Predictions result: {predictions_list}")
        
        return PredictResponse(predictions=predictions_list, version=version, latency=duration)
        
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to preserve correct status codes
        raise http_exc
    except Exception as e:
        logger.error(f"Error during inference pipeline execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during prediction: {str(e)}"
        )