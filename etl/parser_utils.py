import json
import pandas as pd
import h3
from collections import Counter
from google.cloud import storage
import etl_utils
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def parse_taxi_blobs_gcp(bucket_name, h3_to_region_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Parse taxi blobs from GCP bucket."""
    logging.info("Parsing taxi blobs from GCP bucket")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    records = []
    h3_to_region = h3_to_region_df.set_index("h3_id")["region"].to_dict()
    unknown_h3 = set()

    for blob in bucket.list_blobs(prefix="taxi/"):
        filename = blob.name.split("/")[-1]
        if filename.endswith(".json") and blob.time_created.replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
            js = json.loads(blob.download_as_text())["features"][0]
            reading_time = js["properties"]["timestamp"]
            coord_list = js["geometry"]["coordinates"]
            taxi_counter = Counter()
            for coord in coord_list:
                lat, lon = coord[1], coord[0]
                h3_cell = h3.latlng_to_cell(lat, lon, 7)
                if h3_cell in h3_to_region:
                    region = h3_to_region[h3_cell]
                    taxi_counter[region] += 1
                else:
                    unknown_h3.add(h3_cell)
            if taxi_counter:
                for region, count in taxi_counter.items():
                    records.append({
                        "reading_time": reading_time,
                        "region": region,
                        "num_taxis": count
                    })
    if unknown_h3:
        logging.error(f"INCIDENT: Skipped unknown h3s: {sorted(unknown_h3)}")
    df = pd.DataFrame(records)
    if not df.empty:
        df['reading_time'] = pd.to_datetime(df['reading_time'])
        df['reading_time'] = etl_utils.ceil_dt_to_5min(df['reading_time'])
    logging.info(f"Taxi records parsed: {len(df)} rows")
    return df

def parse_current_weather_blobs_gcp(bucket_name, station_to_region_df, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Parse weather blobs from GCP bucket."""
    logging.info("Parsing weather blobs from GCP bucket")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    patterns = {
        "rain":   "rain/",
        "temp":   "temp/",
        "humid":  "humid/"
    }
    
    reading_formatted_list = []

    for category, prefix in patterns.items():
        for blob in bucket.list_blobs(prefix=prefix):
            filename = blob.name.split("/")[-1]
            if filename.endswith(".json") and blob.time_created.replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
                js = json.loads(blob.download_as_text())["data"]["readings"][0]
                reading_time = js['timestamp']
                reading_list = js['data']
                for reading in reading_list:
                    reading_formatted_list.append({
                        'station_id': reading['stationId'],
                        'reading_time': reading_time,
                        'variable': category,
                        'value': reading['value']
                    })
    df = pd.DataFrame(reading_formatted_list)
    if not df.empty:
        df = df.merge(station_to_region_df, on="station_id", how="left")
        df['reading_time'] = pd.to_datetime(df['reading_time'])
        df['reading_time'] = etl_utils.ceil_dt_to_5min(df['reading_time'])
        missing = df[df["region"].isna()]
        if not missing.empty:
            logging.error(f"INCIDENT: Unmapped stations: {missing['station_id'].unique()}")
        df_pivot = (
            df.pivot_table(index=["reading_time", "region"],
                           columns="variable",
                           values="value",
                           aggfunc="mean")
            .reset_index()
        )
        df_pivot['reading_time'] = pd.to_datetime(df_pivot['reading_time'])
        logging.info(f"Weather records parsed: {len(df_pivot)} rows")
        return df_pivot
    logging.warning("No weather records found")
    return pd.DataFrame()

def parse_forecast_blobs_gcp(bucket_name, area_to_region_df, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Parse forecast blobs from GCP bucket."""
    logging.info("Parsing forecast blobs from GCP bucket")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    area_to_region = area_to_region_df.set_index('name')['region'].to_dict()
    unknown_areas = set()
    records = []

    for blob in bucket.list_blobs(prefix="forecast/"):
        filename = blob.name.split("/")[-1]
        if filename.endswith(".json") and blob.time_created.replace(tzinfo=None) >= cutoff.replace(tzinfo=None):
            js = json.loads(blob.download_as_text())["data"]["items"][0]
            reading_time = cutoff  # Use cutoff as reading_time
            for fcast in js["forecasts"]:
                area = fcast["area"]
                if area in area_to_region:
                    records.append((
                        reading_time,
                        area_to_region[area],
                        fcast["forecast"]
                    ))
                else:
                    unknown_areas.add(area)
    if unknown_areas:
        logging.error(f"INCIDENT: Skipped unknown areas in: {sorted(unknown_areas)}")
    df = pd.DataFrame(records, columns=["reading_time", "region", "forecast"])
    if not df.empty:
        df["reading_time"] = pd.to_datetime(df["reading_time"])
        all_forecast_types = ['Fair', 'Fair (Day)', 'Fair (Night)', 'Fair and Warm', 'Partly Cloudy', 'Partly Cloudy (Day)', 'Partly Cloudy (Night)', 'Cloudy', 'Hazy', 'Slightly Hazy', 'Windy', 'Mist', 'Fog', 'Light Rain', 'Moderate Rain', 'Heavy Rain', 'Passing Showers', 'Light Showers', 'Showers', 'Heavy Showers', 'Thundery Showers', 'Heavy Thundery Showers', 'Heavy Thundery Showers with Gusty Winds']
        fcount  = df.groupby(["reading_time", "region", "forecast"], observed=True).size()
        prop = fcount / fcount.groupby(level=[0, 1]).sum()
        df_pivot = (
            prop.unstack(fill_value=0)
            .reindex(columns=all_forecast_types, fill_value=0)
            .reset_index()
        )
        df_pivot = df_pivot.rename(columns={
            'Fair (Day)': 'Fair Day',
            'Fair (Night)': 'Fair Night',
            'Partly Cloudy (Day)': 'Partly Cloudy Day',
            'Partly Cloudy (Night)': 'Partly Cloudy Night'            
        })
        logging.info(f"Forecast records parsed: {len(df_pivot)} rows")
        return df_pivot
    logging.warning("No forecast records found")
    return pd.DataFrame()


