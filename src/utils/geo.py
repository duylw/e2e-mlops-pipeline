import os
import zipfile

import numpy as np
import pandas as pd


def download_taxi_zones(
    url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip",
    zip_path: str = "taxi_zones.zip",
    extract_to: str = "taxi_zones_data",
) -> None:
    """Download and extract NYC TLC taxi zone shapefiles."""
    import requests

    if os.path.exists(extract_to):
        print(f"Directory {extract_to} already exists. Skipping download.")
        return

    print(f"Downloading taxi zones from {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as file:
        file.write(response.content)

    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    if os.path.exists(zip_path):
        os.remove(zip_path)
    print("Download and extraction completed successfully.")


def create_coordinate_lookup(shapefile_path: str) -> pd.DataFrame:
    """Create a LocationID -> centroid coordinates lookup table."""
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path)
    centroids_gps = gdf.geometry.centroid.to_crs("EPSG:4326")
    gdf["longitude"] = centroids_gps.x
    gdf["latitude"] = centroids_gps.y

    lookup = pd.DataFrame(gdf[["LocationID", "borough", "zone", "latitude", "longitude"]])
    return lookup.rename(columns={"borough": "Borough", "zone": "Zone"})


def calculate_haversine(lat1, lon1, lat2, lon2, earth_radius: float = 3958.8):
    """Calculate haversine distance in miles by default."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return earth_radius * 2 * np.arcsin(np.sqrt(a))
