from pydantic import BaseModel, Field
from typing import List, Optional

class TripInput(BaseModel):
    VendorID: int = Field(..., description="ID of the taxi vendor (1 = Creative Mobile Technologies, 2 = VeriFone Inc)", examples=[2])
    lpep_pickup_datetime: str = Field(..., description="Pickup datetime in YYYY-MM-DD HH:MM:SS format", examples=["2026-01-15 08:30:00"])
    PULocationID: int = Field(..., description="Pickup TLC Taxi Zone LocationID", examples=[74])
    DOLocationID: int = Field(..., description="Dropoff TLC Taxi Zone LocationID", examples=[42])
    passenger_count: Optional[float] = Field(None, description="Number of passengers", examples=[1.0])
    trip_type: Optional[float] = Field(None, description="Trip type (1.0 = Street-hail, 2.0 = Dispatch)", examples=[1.0])

class PredictRequest(BaseModel):
    trips: List[TripInput]

class PredictResponse(BaseModel):
    predictions: List[float] = Field(..., description="List of estimated trip durations in minutes")
    version: int = Field(..., description="Model version")
    latency: float = Field(..., description="Prediction latency in seconds")