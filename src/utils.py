import os
import zipfile
import requests
import pandas as pd
import geopandas as gpd
import numpy as np

def download_taxi_zones(url='https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip', zip_path='taxi_zones.zip', extract_to='taxi_zones_data'):
    """
    Downloads and extracts NYC Taxi Zone shapefiles if they do not exist.
    """
    if os.path.exists(extract_to):
        print(f"Directory {extract_to} already exists. Skipping download.")
        return
        
    print(f"Downloading taxi zones from {url}...")
    r = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(r.content)
    
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print("Download and extraction completed successfully.")

def create_coordinate_lookup(shapefile_path):
    """
    Reads shapefiles, calculates precise zone centroids in flat system,
    converts centroids to GPS coordinates (EPSG:4326), and returns a lookup DataFrame.
    """
    gdf = gpd.read_file(shapefile_path)
    centroids = gdf.geometry.centroid
    centroids_gps = centroids.to_crs("EPSG:4326")
    gdf['longitude'] = centroids_gps.x
    gdf['latitude'] = centroids_gps.y

    df_lookup = pd.DataFrame(gdf[['LocationID', 'borough', 'zone', 'latitude', 'longitude']])
    df_lookup.rename(columns={'borough': 'Borough', 'zone': 'Zone'}, inplace=True)

    return df_lookup

def calculate_haversine(lat1, lon1, lat2, lon2, earth_radius=3958.8):
    """
    Calculates the haversine distance between two sets of GPS points.
    earth_radius: Default is 3958.8 miles. Use 6371.0 for kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = earth_radius * c
    return distance
