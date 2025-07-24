import pandas as pd
from google.cloud import bigquery, storage
from io import StringIO
import logging
import numpy as np
import os
import pickle
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def load_csv_from_gcs(bucket_name, blob_name):
    """Load a CSV file from GCS bucket into a pandas DataFrame."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_text()
        logging.info(f"Loaded {blob_name} from bucket {bucket_name}")
        return pd.read_csv(StringIO(data))
    except Exception as e:
        logging.error(f"INCIDENT: Failed to load {blob_name} from {bucket_name}: {e}")
        raise

def load_model_from_gcs(bucket_name, model_path="models/xgboost_model_2h.pkl"):
    """Load the XGBoost model from Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_path)
        
        # Download to a temporary file
        temp_file = "/tmp/temp_model.pkl"
        blob.download_to_filename(temp_file)
        
        # Load the model
        with open(temp_file, "rb") as f:
            model = pickle.load(f)
        
        # Clean up
        os.remove(temp_file)
        
        logging.info(f"Successfully loaded model from gs://{bucket_name}/{model_path}")
        return model
    except Exception as e:
        logging.error(f"INCIDENT: Failed to load model from GCS: {str(e)}")
        raise

def load_prophet_base_forecasts_from_gcs(bucket_name, folder_path):
    """
    Load prophet base forecasts from GCS bucket into a dictionary of pandas DataFrames.
    
    Args:
        bucket_name (str): The name of the GCS bucket
        folder_path (str): The folder path within the bucket where prophet forecast files are stored
        
    Returns:
        dict: Dictionary with region names as keys and prophet forecasts as DataFrames
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Initialize the dictionary to store the forecasts
        prophet_base_forecasts = {}
        
        # The region names to look for
        regions = ['Central', 'North', 'East', 'West']
        
        for region in regions:
            # Construct the blob name for this region
            blob_name = f"{folder_path}{region}.csv"
            
            # Get the blob
            blob = bucket.blob(blob_name)
            
            # Check if the blob exists
            if not blob.exists():
                logging.warning(f"No prophet base forecast file found for region {region} at {blob_name}")
                continue
                
            # Download the CSV data
            data = blob.download_as_text()
            logging.info(f"Loaded {blob_name} from bucket {bucket_name}")
            
            # Convert to DataFrame
            df = pd.read_csv(StringIO(data))
            
            # Convert 'ds' column to datetime with UTC timezone
            if 'ds' in df.columns:
                df['ds'] = pd.to_datetime(df['ds'], utc=True)
            else:
                logging.warning(f"No 'ds' column found in prophet forecast for region {region}")
                continue
            
            # Store in the dictionary
            prophet_base_forecasts[region] = df
        
        if not prophet_base_forecasts:
            logging.error("INCIDENT: No prophet base forecasts were loaded successfully")
            raise ValueError("No prophet base forecasts could be loaded")
            
        logging.info(f"Successfully loaded prophet base forecasts for {list(prophet_base_forecasts.keys())}")
        return prophet_base_forecasts
        
    except Exception as e:
        logging.error(f"INCIDENT: Failed to load prophet base forecasts from {bucket_name}/{folder_path}: {e}")
        raise

def ceil_dt_to_5min(dt_series: pd.Series) -> pd.Series:
    """Round up pandas datetime series to the next 5-minute interval."""
    return (dt_series + pd.Timedelta(minutes=4)).dt.floor('5min')

def get_current_execution_ts() -> pd.Timestamp:
    """Rounds current UTC time to the next 5-minute interval."""
    now = pd.Timestamp.utcnow()
    return now.floor('5min')

def join_records_dim(readings_df: pd.DataFrame, taxi_df: pd.DataFrame, forecast_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merge readings, taxi, and optionally forecast DataFrames on ['reading_time', 'region'].
    Uses outer join to retain all records.
    
    Handles None inputs by returning any available DataFrame.
    Returns empty DataFrame if all inputs are None.
    """
    try:
        # Handle cases where inputs may be None
        if readings_df is None and taxi_df is None and forecast_df is None:
            logging.warning("All input DataFrames are None. Returning empty DataFrame.")
            return pd.DataFrame(columns=["reading_time", "region"])
        
        # If only one DataFrame is available, return it
        if readings_df is None and taxi_df is None:
            logging.warning("readings_df and taxi_df are None. Returning only forecast_df.")
            return forecast_df
        elif readings_df is None and forecast_df is None:
            logging.warning("readings_df and forecast_df are None. Returning only taxi_df.")
            return taxi_df
        elif taxi_df is None and forecast_df is None:
            logging.warning("taxi_df and forecast_df are None. Returning only readings_df.")
            return readings_df
        
        # Handle two-way merges when one DataFrame is None
        if readings_df is None:
            logging.warning("readings_df is None. Merging taxi_df and forecast_df.")
            return pd.merge(taxi_df, forecast_df, on=["reading_time", "region"], how="outer")
        elif taxi_df is None:
            logging.warning("taxi_df is None. Merging readings_df and forecast_df.")
            return pd.merge(readings_df, forecast_df, on=["reading_time", "region"], how="outer")
        elif forecast_df is None:
            logging.warning("forecast_df is None. Merging readings_df and taxi_df.")
            return pd.merge(readings_df, taxi_df, on=["reading_time", "region"], how="outer")
        
        # Standard case - all DataFrames available
        final_df = pd.merge(readings_df, taxi_df, on=["reading_time", "region"], how="outer")
        final_df = final_df.merge(forecast_df, on=["reading_time", "region"], how="outer")
        logging.info(f"Merged records shape: {final_df.shape}")
        return final_df
    except Exception as e:
        logging.error(f"INCIDENT: Failed to merge records: {e}")
        # Return empty DataFrame instead of raising
        logging.warning("Returning empty DataFrame due to merge failure")
        return pd.DataFrame(columns=["reading_time", "region"])

def pull_recent_data_bq(bq_client: bigquery.Client, table_id: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Pulls recent data from BigQuery table from cutoff minus 4 hours onward.
    Uses the BigQuery Storage API if available.
    """
    # bq_cutoff set to 4 hours before current cutoff execution time
    bq_cutoff = (cutoff - pd.Timedelta(hours=4)).isoformat()
    try:
        query = f"""
            SELECT * FROM `{table_id}`
            WHERE reading_time >= TIMESTAMP('{bq_cutoff}')
        """
        query_job = bq_client.query(query)
        
        # Try to use the Storage API first, with a fallback to REST
        try:
            # Use Storage API with optimized dtypes
            df = query_job.to_dataframe(create_bqstorage_client=True)
            logging.info("Used BigQuery Storage API for data fetch")
        except ImportError:
            # Fallback to REST API
            df = query_job.to_dataframe()
            logging.info("Used BigQuery REST API for data fetch (Storage API not available)")
        
        # Ensure reading_time is properly typed
        df['reading_time'] = pd.to_datetime(df['reading_time'])
        logging.info(f"Pulled {len(df)} rows from BigQuery table {table_id}")
        return df
    except Exception as e:
        logging.error(f"INCIDENT: Failed to pull recent data from BigQuery: {e}")
        raise

def overwrite_recent_values_with_time_regressors(
    bq_df: pd.DataFrame, new_df: pd.DataFrame, time_dim: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """
    Overwrites non-null values in bq_df using new_df (per-cell basis).
    Clears bq_df values forward from the earliest time a new value appears per column.
    Adds new rows from new_df to bq_df when reading_time is not present.
    Adds time regressors to any newly added rows.
    Ensures the result contains entries for the current cutoff timestamp.
    
    Args:
        bq_df: DataFrame from BigQuery
        new_df: New DataFrame to merge in
        time_dim: Time dimension DataFrame with time regressors
        cutoff: Current timestamp that must be present in the final DataFrame
    """
    try:
        # Make copies to avoid modifying originals
        bq_df = bq_df.copy()
        new_df = new_df.copy()

        # In case new_df is empty initialise with a placeholder DataFrame
        if new_df.empty:
            # Create a placeholder Dataframe with just the current cutoff for all regions
            new_df = pd.DataFrame({
                'reading_time': [cutoff] * 4,
                'region': ['Central', 'East', 'North', 'West']
            })
        # Check if the BigQuery DataFrame is empty
        if bq_df.empty:
            logging.info("BigQuery DataFrame is empty. Using all new data with matching schema.")

            # Use all new data with time regressors
            result_df = new_df.merge(time_dim, on=["reading_time", "region"], how="left")

            # Get the column structure from bq_df (even though it's empty, it should have column definitions)
            bq_columns = set(bq_df.columns)
            
            # Add any missing columns from bq_df schema and fill with nan
            missing_columns = bq_columns - set(result_df.columns)
            for col in missing_columns:
                result_df[col] = np.nan

            # Important: reorder columns to match the original bq_df column order
            result_df = result_df.reindex(columns=bq_df.columns)

            logging.info(f"Created new DataFrame from new data, shape: {result_df.shape}")
            return result_df
        
        # Replace pd.NA with np.nan in numeric columns to avoid NAType errors
        for df in [bq_df, new_df]:
            for col in df.select_dtypes(include=['number']).columns:
                if pd.isna(df[col]).any():
                    df[col] = df[col].fillna(np.nan)

        # Check for duplicates in indices before any operations
        if new_df.duplicated(subset=["reading_time", "region"]).any():
            logging.warning("Duplicates found in new_df. Taking the last occurrence.")
            new_df = new_df.drop_duplicates(subset=["reading_time", "region"], keep="last")
            
        if bq_df.duplicated(subset=["reading_time", "region"]).any():
            logging.warning("Duplicates found in bq_df. Taking the last occurrence.")
            bq_df = bq_df.drop_duplicates(subset=["reading_time", "region"], keep="last")
            
        # Continue with the regular processing if bq_df is not empty
        new_df = new_df.set_index(["reading_time", "region"])
        bq_df = bq_df.set_index(["reading_time", "region"])
        
        # Step 1: Determine earliest timestamp where new data exists per column
        min_times = {}
        for col in new_df.columns:
            if col in bq_df.columns:
                non_null_times = new_df.index.get_level_values("reading_time")[~pd.isna(new_df[col])]
                if not non_null_times.empty:
                    min_times[col] = non_null_times.min()
        
        # Step 2: Drop stale values in bq_df from min_time onward
        bq_df = bq_df.copy()
        for col, min_time in min_times.items():
            if col in bq_df.columns:
                mask = bq_df.index.get_level_values("reading_time") >= min_time
                bq_df.loc[mask, col] = np.nan
        
        # Step 3: Overwrite values in overlapping rows/columns
        overlapping_rows = bq_df.index.intersection(new_df.index)
        overlapping_cols = bq_df.columns.intersection(new_df.columns)
        for col in overlapping_cols:
            non_null_mask = ~pd.isna(new_df.loc[overlapping_rows, col])
            bq_df.loc[overlapping_rows, col] = bq_df.loc[overlapping_rows, col].where(
                ~non_null_mask,
                new_df.loc[overlapping_rows, col]
            )
        
        # Step 4: Find new rows from new_df not present in bq_df, and beyond max_ts
        if not bq_df.empty:
            max_ts = bq_df.index.get_level_values("reading_time").max()
            new_rows_index = new_df.index.difference(bq_df.index)
            new_rows = new_df.loc[new_rows_index]
            # Only filter by max_ts if there's a valid timestamp
            if not pd.isna(max_ts):
                new_rows = new_rows[new_rows.index.get_level_values("reading_time") > max_ts]
        else:
            # If bq_df became empty during processing, use all new rows
            new_rows = new_df
            
        # Step 5: Add time regressors and append new rows
        new_rows = new_rows.reset_index()
        bq_df = bq_df.reset_index()
        
        # Merge time dimensions to new rows
        if not new_rows.empty:
            # Add in rows including the cutoff in case it is not present
            if not (new_rows['reading_time'] == cutoff).any():
                cutoff_row = pd.DataFrame({
                    'reading_time': [cutoff] * 4,
                    'region': ['Central', 'East', 'North', 'West']
                })
                new_rows = pd.concat([new_rows, cutoff_row], ignore_index=True)
                logging.info(f"Added cutoff {cutoff} rows")
            
            new_rows = new_rows.merge(time_dim, on=["reading_time", "region"], how="left")
            
            # Ensure column alignment before concatenation
            bq_columns = set(bq_df.columns)

            # Add any missing columns from bq_df schema and fill with None instead of pd.NA
            missing_columns = bq_columns - set(new_rows.columns)
            for col in missing_columns:
                if col in bq_df.select_dtypes(include=['number']).columns:
                    new_rows[col] = np.nan  # Use np.nan for numeric columns
                else:
                    new_rows[col] = None  # Use None for non-numeric columns
                
            # Ensure new_rows has the same columns in the same order as bq_df
            new_rows = new_rows.reindex(columns=bq_df.columns)
            
            # Handle pd.NA values in new_rows
            for col in bq_df.columns:
                # Only handle columns that exist in new_rows
                if col not in new_rows.columns:
                    continue
                    
                # Handle numeric columns to avoid NAType errors
                if pd.api.types.is_numeric_dtype(bq_df[col].dtype):
                    try:
                        # Convert to float64 to handle NaN values safely
                        bq_df[col] = bq_df[col].astype('float64')
                        new_rows[col] = new_rows[col].astype('float64')
                    except Exception as e:
                        logging.warning(f"Could not convert column {col} to float64: {e}")
            
            # Standard concat with original parameters
            result_df = pd.concat([bq_df, new_rows], ignore_index=True)
        else:
            result_df = bq_df
            
        logging.info(f"Overwritten and appended records, final shape: {result_df.shape}")
        return result_df
    except Exception as e:
        logging.error(f"INCIDENT: Failed to overwrite recent values: {e}")
        logging.error(f"Error details: {str(e)}")
        raise

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fills missing values for selected columns, grouped by region.
    Excludes forecast columns from ffill.
    """
    try:
        df = df.sort_values(["region", "reading_time"])
        exclude_cols = ['forecast_halfh', 'forecast_1h', 'forecast_1halfh', 'forecast_2h']
        cols_to_ffill = [col for col in df.columns if col not in exclude_cols + ['region']]
        df[cols_to_ffill] = df.groupby("region", group_keys=False)[cols_to_ffill].transform(lambda g: g.ffill())
        logging.info("Forward-filled missing values")
        return df
    except Exception as e:
        logging.error(f"INCIDENT: Failed to fill missing values: {e}")
        raise

def update_bq_data(cutoff: pd.Timestamp, bq_client: bigquery.Client, records_table_id: str, records_df: pd.DataFrame, forecast_table_id: str = None, forecasts: pd.DataFrame = None):
    """
    Update BigQuery tables with new data and forecasts.
    
    Args:
        cutoff: Timestamp for the current execution, used to determine data to delete
        bq_client: BigQuery client
        records_table_id: Main table ID to update
        records_df: DataFrame with updated records
        forecast_table_id: Forecast table ID (optional)
        forecasts: DataFrame with forecast records (optional)

    """
    try:
        # Prepare main table data
        records_df = _prepare_dataframe_for_bq(records_df, bq_client, records_table_id)
        temp_table = records_table_id + "_staging"
        _upload_dataframe_to_staging(bq_client, records_df, temp_table)
        
        # Calculate cutoff for data deletion (4 hours before execution time)
        bq_cutoff = (cutoff - pd.Timedelta(hours=4)).isoformat()

        # Prepare SQL transaction
        sql_parts = []
        query_params = [
            bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", bq_cutoff)
        ]
        
        # Add main table update SQL
        column_names = [col for col in records_df.columns]
        columns_sql = ", ".join([f"`{col}`" for col in column_names])
        
        sql_parts.append(f"""
            -- Update main table
            DELETE FROM {records_table_id}
            WHERE reading_time >= TIMESTAMP(@cutoff);
            
            INSERT INTO {records_table_id} ({columns_sql})
            SELECT {columns_sql} FROM {temp_table};
        """)
        
        # Handle forecasts if provided
        if forecast_table_id is not None and forecasts is not None and not forecasts.empty:
            # Prepare forecast data
            forecasts = forecasts.copy()
            if 'created_at' not in forecasts.columns:
                forecasts['created_at'] = cutoff
                
            forecasts = _prepare_timestamp_columns(forecasts, ['timestamp', 'created_at'])
            
            # Upload to staging
            forecast_temp_table = forecast_table_id + "_staging"
            _upload_dataframe_to_staging(bq_client, forecasts, forecast_temp_table)
            
            # Add forecast table replacement SQL
            forecast_columns = forecasts.columns.tolist()
            forecast_columns_sql = ", ".join([f"`{col}`" for col in forecast_columns])
            
            sql_parts.append(f"""
                -- Complete replacement of the forecast table
                DELETE FROM {forecast_table_id} WHERE 1=1;
                
                INSERT INTO {forecast_table_id} ({forecast_columns_sql})
                SELECT {forecast_columns_sql} FROM {forecast_temp_table};
            """)
        
        # Execute transaction
        sql = "BEGIN TRANSACTION;\n" + "\n".join(sql_parts) + "\nCOMMIT TRANSACTION;"
        
        job_config = bigquery.QueryJobConfig(query_parameters=query_params)
        job = bq_client.query(sql, job_config=job_config)
        job.result()  # Wait for query job to complete
        
        logging.info(f"Updated BigQuery table {records_table_id}" + 
                    (f" and replaced all forecasts in {forecast_table_id}" if forecast_table_id else ""))
            
    except Exception as e:
        logging.error(f"INCIDENT: Failed to update BigQuery tables: {e}")
        logging.error(f"Error details: {str(e)}")
        raise

def _prepare_dataframe_for_bq(df: pd.DataFrame, bq_client: bigquery.Client, table_id: str) -> pd.DataFrame:
    """Prepare DataFrame with correct types based on BigQuery table schema."""
    df = df.copy()
    
    # Get the table schema
    table = bq_client.get_table(table_id)
    schema = table.schema
    
    # Create a mapping of column names to their BigQuery types
    schema_dict = {field.name: field.field_type for field in schema}
    
    # Convert DataFrame types based on BigQuery schema
    for col in df.columns:
        if col in schema_dict:
            # Convert based on BigQuery type
            if schema_dict[col] in ('INTEGER', 'INT64'):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
            elif schema_dict[col] in ('FLOAT', 'FLOAT64'):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
            elif schema_dict[col] in ('BOOLEAN', 'BOOL'):
                df[col] = df[col].fillna(False).astype('bool')
            elif schema_dict[col] == 'TIMESTAMP':
                # Use utc=True for handling timezone-aware datetimes
                df[col] = pd.to_datetime(df[col], utc=True)
            elif schema_dict[col] == 'STRING':
                df[col] = df[col].fillna('').astype(str)
    
    return df

def _prepare_timestamp_columns(df: pd.DataFrame, timestamp_columns: list) -> pd.DataFrame:
    """Prepare timestamp columns in a DataFrame."""
    df = df.copy()
    for col in timestamp_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df

def _upload_dataframe_to_staging(bq_client: bigquery.Client, df: pd.DataFrame, staging_table: str) -> None:
    """Upload DataFrame to BigQuery staging table."""
    try:
        # Get schema if table exists, otherwise None (let BQ infer)
        schema = None
        try:
            table = bq_client.get_table(staging_table)
            schema = table.schema
        except Exception:
            pass
        
        # Create job config
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE"
        )
        
        # Load DataFrame to staging table
        load_job = bq_client.load_table_from_dataframe(df, staging_table, job_config=job_config)
        load_job.result()  # Wait for the load job to complete
        
        logging.info(f"Loaded {len(df)} rows into staging table {staging_table}")
    except Exception as e:
        logging.error(f"Failed to upload DataFrame to staging table {staging_table}: {e}")
        raise