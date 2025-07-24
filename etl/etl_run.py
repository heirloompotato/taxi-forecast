import pandas as pd
import logging
import os
from flask import Flask, request
from google.cloud import bigquery

import parser_utils
import etl_utils
import forecast_utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/", methods=["POST"])
def run_etl():
    try:
        logging.info("Starting ETL run")
        cutoff = etl_utils.get_current_execution_ts()
        logging.info(f"Cutoff time set to {cutoff}")

        # Set GCP resource names
        raw_json_bucket = os.environ['RAW_JSON_BUCKET']
        config_bucket = os.environ['CONFIG_BUCKET']
        records_table_id = os.environ['BQ_TABLE_ID']
        forecast_table_id = os.environ['BQ_FORECAST_TABLE_ID']

        # Load config tables from GCS
        logging.info("Loading config tables and model from GCS")
        h3_to_region_df = etl_utils.load_csv_from_gcs(config_bucket, "region_dim.csv")
        station_to_region_df = etl_utils.load_csv_from_gcs(config_bucket, "stations_dim.csv")
        area_to_region_df = etl_utils.load_csv_from_gcs(config_bucket, "area_dim.csv")
        time_dim = etl_utils.load_csv_from_gcs(config_bucket, "time_grid.csv")
        time_dim['reading_time'] = pd.to_datetime(time_dim['reading_time'])
        model = etl_utils.load_model_from_gcs(config_bucket, "model/xgboost_model_v3.pkl")
        prophet_base_forecasts = etl_utils.load_prophet_base_forecasts_from_gcs(config_bucket, "prophet_base_forecasts/")
        logging.info("Config tables and model loaded successfully")

        # Parse data from GCS bucket
        logging.info("Parsing weather and taxi records from GCS")
        readings_records = parser_utils.parse_current_weather_blobs_gcp(raw_json_bucket, station_to_region_df, cutoff)
        taxi_records = parser_utils.parse_taxi_blobs_gcp(raw_json_bucket, h3_to_region_df, cutoff)

        # Only parse forecast if minute is 10 or 40
        is_10or40min = cutoff.minute in [10, 40]
        if is_10or40min:
            logging.info("Cutoff is at 10 or 40 min, parsing forecast records")
            forecast_records = parser_utils.parse_forecast_blobs_gcp(raw_json_bucket, area_to_region_df, cutoff)
            logging.info("Merging all records")
            merged_records = etl_utils.join_records_dim(readings_records, taxi_records, forecast_records)

        else:
            logging.info("Cutoff is not at 10 or 40 min, skipping to merging")
            merged_records = etl_utils.join_records_dim(readings_records, taxi_records)

        logging.info(f"Merged records shape: {merged_records.shape}")

        # Query for the latest snapshot from BigQuery
        bq_client = bigquery.Client()
        logging.info("Pulling recent data from BigQuery")
        bq_snapshot = etl_utils.pull_recent_data_bq(bq_client, records_table_id, cutoff)
        logging.info(f"Pulled snapshot shape: {bq_snapshot.shape}")

        # Overwrite recent values and fill missing values
        logging.info("Overwriting recent values and filling missing values")
        overwritten_records = etl_utils.overwrite_recent_values_with_time_regressors(bq_snapshot, merged_records, time_dim, cutoff)
        ffilled_records = etl_utils.fill_missing_values(overwritten_records)
        logging.info(f"Final records shape: {ffilled_records.shape}")

        # Forecast num taxis
        forecasted_records, forecasts = forecast_utils.forecast_num_taxis(
            ffilled_records, 
            prophet_base_forecasts=prophet_base_forecasts,
            model=model,
            execution_ts=cutoff
        )

        # Update BigQuery table (delete + insert)
        logging.info("Updating BigQuery table")
        etl_utils.update_bq_data(cutoff, bq_client, records_table_id, forecasted_records, forecast_table_id, forecasts)
        logging.info("BigQuery table updated successfully")

        return "ETL run completed", 200
    except Exception as e:
        logger.error(f"ETL process failed: {str(e)}")
        return f"Error: {str(e)}", 500

# Health check endpoint
@app.route("/", methods=["GET"])
def health_check():
    return "Service is running", 200

if __name__ == "__main__":
    # Use PORT environment variable provided by Cloud Run
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)