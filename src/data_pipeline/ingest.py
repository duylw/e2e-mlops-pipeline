import os
import zipfile
import requests
import pandas as pd
import geopandas as gpd
import logging
import argparse

logger = logging.getLogger("Ingest Data")
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    prog='Ingest Data',
    description='Ingests taxi data from TLC',
    epilog='Text at the bottom of help'
)
parser.add_argument('--month', type=int, default=1)
parser.add_argument('--year', type=int, default=2026)

def create_coordinate_lookup(shapefile_path):
    """
    Reads shapefiles, calculates precise zone centroids in flat system,
    converts centroids to GPS coordinates (EPSG:4326), and returns a lookup DataFrame.
    """
    # 1. Read shapefile (native state flat coordinate system - feet)
    gdf = gpd.read_file(shapefile_path)

    # 2. Calculate centroids first in the native coordinate system
    centroids = gdf.geometry.centroid

    # 3. Convert centroids projection to GPS coordinate system (Lat/Lon)
    centroids_gps = centroids.to_crs("EPSG:4326")

    # 4. Extract latitude and longitude
    gdf['longitude'] = centroids_gps.x
    gdf['latitude'] = centroids_gps.y

    # 5. Convert to standard pandas DataFrame
    df_lookup = pd.DataFrame(gdf[['LocationID', 'borough', 'zone', 'latitude', 'longitude']])
    df_lookup.rename(columns={'borough': 'Borough', 'zone': 'Zone'}, inplace=True)

    return df_lookup

def download_taxi_zones(url='https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip', zip_path='taxi_zones.zip', extract_to='taxi_zones_data'):
    """
    Downloads and extracts NYC Taxi Zone shapefiles if they do not exist.
    """
    if os.path.exists(extract_to):
        logger.info(f"Directory {extract_to} already exists. Skipping download.")
        return
        
    logger.info(f"Downloading taxi zones from {url}...")
    r = requests.get(url)
    with open(zip_path, 'wb') as f:
        f.write(r.content)
    
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    # Clean up zip file
    if os.path.exists(zip_path):
        os.remove(zip_path)
    logger.info("Download and extraction completed successfully.")

def download_green_taxi_data(month, year, url, base_path):
    file_name = f"green_tripdata_{year}-{month:02d}.parquet"
    full_path = os.path.join(base_path, file_name) 
    
    if os.path.exists(full_path):
        logger.info(f"File {full_path} already exists. Skipping download.")
        return
    
    logger.info(f"Downloading taxi zones from {url}...")
    r = requests.get(url)
    with open(full_path, 'wb') as f:
        f.write(r.content)
    logger.info("Download and extraction completed successfully.")

if __name__ == '__main__':

    args = parser.parse_args()
    month = args.month
    year = args.year

    logger.info("Ingesting taxi data...")
    logger.info("Downloading Taxi Zone...")

    # 1. Download and generate zone lookup with coordinates
    lookup_csv = 'data/processed/nyc_taxi_lookup_with_coords.csv'
    if not os.path.exists(lookup_csv):
        logger.info(f"Lookup file {lookup_csv} not found. Preparing coordinates table...")
        download_taxi_zones(
            url='https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip',
            zip_path='data/raw/taxi_zones.zip',
            extract_to='data/raw/taxi_zones_data'
        )
        # Locate the shapefile inside extracted directory
        shapefile = 'data/raw/taxi_zones_data/taxi_zones/taxi_zones.shp'
        if not os.path.exists(shapefile):
            import glob
            shp_files = glob.glob('data/raw/taxi_zones_data/**/*.shp', recursive=True)
            if shp_files:
                shapefile = shp_files[0]
            else:
                raise FileNotFoundError("Could not find TLC shapefile (.shp) in taxi_zones_data.")
                
        df_lookup_with_loc = create_coordinate_lookup(shapefile)
        df_lookup_with_loc.to_csv(lookup_csv, index=False)
        logger.info(f"Created coordinate lookup table and saved to {lookup_csv}.")
    else:
        logger.info(f"Loading existing lookup table from {lookup_csv}.")
        df_lookup_with_loc = pd.read_csv(lookup_csv)

    logger.info(f"Downloading Green Taxi Data for {month}/{year}...")
    # 2. Download Green Taxi Data for {month}/{year}
    download_green_taxi_data(
        month=month,
        year=year,
        url=f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet',
        base_path='data/raw'
    )
    logger.info('Download completed.')    