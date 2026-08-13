from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class TripInput(BaseModel):
    VendorID: int = Field(..., description="Taxi vendor id", examples=[2])
    lpep_pickup_datetime: str = Field(
        ...,
        description="Pickup datetime in YYYY-MM-DD HH:MM:SS format",
        examples=["2026-01-15 08:30:00"],
    )
    PULocationID: int = Field(..., description="Pickup TLC Taxi Zone LocationID", examples=[74])
    DOLocationID: int = Field(..., description="Dropoff TLC Taxi Zone LocationID", examples=[42])
    passenger_count: Optional[float] = Field(None, description="Number of passengers", examples=[1.0])
    trip_type: Optional[float] = Field(None, description="Trip type: 1.0 street-hail, 2.0 dispatch", examples=[1.0])

    @field_validator("lpep_pickup_datetime")
    @classmethod
    def validate_pickup_datetime(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise ValueError("must use YYYY-MM-DD HH:MM:SS format") from exc
        return value


class PredictRequest(BaseModel):
    trips: List[TripInput]


class PredictResponse(BaseModel):
    predictions: List[float] = Field(..., description="Estimated trip durations in minutes")
    model_name: str
    model_version: int
    model_alias: str
    latency: float = Field(..., description="Prediction latency in seconds")


class MetadataResponse(BaseModel):
    model_name: str
    model_version: int
    model_alias: str
    model_loaded: bool
