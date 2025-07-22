import pandas as pd
import numpy as np
import logging
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def _prepare_xgboost_data(df):
    """Prepare data for XGBoost model including all regions."""
    # Create a copy to avoid modifying the original
    xgb_df = df.copy()

    # Convert num_taxis to float explicitly
    xgb_df['num_taxis'] = xgb_df['num_taxis'].astype(float)
    # Ensure reading_time is timezone aware with UTC
    xgb_df['reading_time'] = pd.to_datetime(xgb_df['reading_time'], utc=True)
    
    # One-hot encode the region
    region_dummies = pd.get_dummies(xgb_df['region'], prefix='region')
    xgb_df = pd.concat([xgb_df, region_dummies], axis=1)
    
    # Create lag features (previous taxi counts)
    for lag in [1, 2, 3, 6, 12]:  # 5, 10, 15, 30, 60 minutes ago
        xgb_df[f'taxi_lag_{lag}'] = xgb_df.groupby('region')['num_taxis'].shift(lag)
    
    # Create rolling mean features
    for window in [4, 12, 24, 48]:  # 20min, 1h, 2h, 4h windows
        xgb_df[f'taxi_rolling_{window}'] = xgb_df.groupby('region')['num_taxis'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return xgb_df

def _apply_smoothing(forecast_df):
    """Apply exponential smoothing to predictions to reduce zigzags."""
    for region in forecast_df['region_name'].unique():
        region_mask = forecast_df['region_name'] == region
        region_values = forecast_df.loc[region_mask, 'predicted_value'].values
        
        # Apply exponential smoothing
        alpha = 0.7
        smoothed = [region_values[0]]
        for i in range(1, len(region_values)):
            smoothed.append(alpha * region_values[i] + (1-alpha) * smoothed[i-1])
            
        forecast_df.loc[region_mask, 'predicted_value'] = smoothed
    
    return forecast_df

def _add_dynamic_confidence_intervals(forecast_df):
    """Add dynamic confidence intervals that widen over time."""
    for region in forecast_df['region_name'].unique():
        region_mask = forecast_df['region_name'] == region
        forecast_region = forecast_df[region_mask].sort_values('timestamp')
        
        # Get number of timestamps for this region
        n_timestamps = len(forecast_region)
        
        # Create increasing factors (1.0 to 2.0)
        factors = np.linspace(1.0, 2.0, n_timestamps)
        
        # Calculate base width
        base_std = forecast_region['predicted_value'].std() * 0.5 if len(forecast_region) > 1 else 10
        center = forecast_region['predicted_value']
        
        # Recalculate bounds with widening factors
        new_lower = center - 1.96 * base_std * factors
        new_upper = center + 1.96 * base_std * factors
        
        # Update the DataFrame
        forecast_df.loc[region_mask, 'lower_bound_95'] = new_lower.values
        forecast_df.loc[region_mask, 'upper_bound_95'] = new_upper.values
    
    return forecast_df

def _forecast_future_with_xgboost(model, xgb_df, features, time_dim, future_periods=24):
    """Generate forecasts for future periods using XGBoost."""
    all_forecasts = []
    
    for region in ['Central', 'North', 'East', 'West']:
        # Only select data for this region and ensure unique index
        region_df = xgb_df[xgb_df['region'] == region].copy()
        region_df = region_df.reset_index(drop=True)
        
        if len(region_df) == 0:
            logging.warning(f"No data available for region {region}, skipping forecast")
            continue
        
        # Start with the last known datapoints
        future_rows = []
        last_known = region_df.iloc[-future_periods:].copy() if len(region_df) >= future_periods else region_df.copy()
        
        # Create future timestamps
        last_time = region_df['reading_time'].max()
        future_times = pd.date_range(
            start=last_time + pd.Timedelta(minutes=5), 
            periods=future_periods, 
            freq='5min',
            tz='UTC'
        )
        
        # Create a dataframe with future timestamps
        for i, future_time in enumerate(future_times):
            # Use the base row from last_known
            new_row = last_known.iloc[i % len(last_known)].copy()
            new_row['reading_time'] = future_time
            
            # Get time regressors from time_dim
            time_regressors = time_dim[time_dim['reading_time'] == future_time]
            if len(time_regressors) > 0:
                # Copy time regressors from time_dim
                for col in ['minute', 'hour', 'hour_decimal', 'dayofweek', 'is_weekend', 
                           'is_surcharge_hour', 'time_sin', 'time_cos']:
                    if col in time_regressors.columns:
                        new_row[col] = time_regressors.iloc[0][col]
            else:
                # If not found in time_dim, calculate them
                new_row['minute'] = future_time.minute
                new_row['hour'] = future_time.hour
                new_row['hour_decimal'] = future_time.hour + future_time.minute / 60
                new_row['dayofweek'] = future_time.dayofweek
                new_row['is_weekend'] = 1 if future_time.dayofweek >= 5 else 0
                
                # Time cyclical features
                minutes_in_day = future_time.hour * 60 + future_time.minute
                new_row['time_sin'] = np.sin(2 * np.pi * minutes_in_day / (24 * 60))
                new_row['time_cos'] = np.cos(2 * np.pi * minutes_in_day / (24 * 60))
                
                # Surcharge hours
                is_surcharge = (future_time.hour >= 0 and future_time.hour < 6) or \
                               (future_time.hour >= 18 and future_time.hour < 24)
                new_row['is_surcharge_hour'] = 1 if is_surcharge else 0

            future_rows.append(new_row)
        
        # Create future dataframe
        future_df = pd.DataFrame(future_rows).reset_index(drop=True)
        
        # Make forecasts iteratively
        for i in range(future_periods):
            if i > 0:
                # Update lag features based on previous predictions
                for lag in [1, 2, 3, 6, 12]:
                    lag_idx = i - lag
                    if lag_idx >= 0:
                        future_df.loc[i, f'taxi_lag_{lag}'] = future_df.loc[lag_idx, 'num_taxis']
                    else:
                        if abs(lag_idx) < len(region_df):
                            future_df.loc[i, f'taxi_lag_{lag}'] = region_df.iloc[lag_idx]['num_taxis']
            
            # Update rolling mean features
            for window in [4, 12, 24, 48]:
                values_to_consider = []
                for w in range(window):
                    w_idx = i - w
                    if w_idx >= 0:
                        values_to_consider.append(future_df.iloc[w_idx]['num_taxis'])
                    else:
                        if abs(w_idx) < len(region_df):
                            values_to_consider.append(region_df.iloc[w_idx]['num_taxis'])
                future_df.loc[i, f'taxi_rolling_{window}'] = np.mean(values_to_consider) if values_to_consider else np.nan
            
            # Make prediction
            X_future = future_df.iloc[i:i+1][features].copy()
            if X_future.isna().any().any():
                X_future = X_future.fillna(0)
                
            y_pred = model.predict(X_future)[0]
            future_df.loc[i, 'num_taxis'] = y_pred
            
        # Prepare forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': future_df['reading_time'],
            'region_name': region,
            'predicted_value': future_df['num_taxis'],
            'model_version': 'xgboost_v1'
        })
        
        all_forecasts.append(forecast_df)
    
    # Combine all forecasts
    if all_forecasts:
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        
        # Add confidence intervals
        combined_forecasts = _add_dynamic_confidence_intervals(combined_forecasts)
        
        # Apply smoothing
        combined_forecasts = _apply_smoothing(combined_forecasts)
        
        return combined_forecasts
    else:
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['timestamp', 'region_name', 'predicted_value', 
                                    'lower_bound_95', 'upper_bound_95', 'model_version'])

def forecast_num_taxis(df, time_dim, model, execution_ts=None, horizon=24):
    """
    Generate forecasts for the next 2 hours (24 5-minute periods).
    
    Args:
        df: DataFrame with current data
        time_dim: DataFrame with time dimension data
        execution_ts: The execution timestamp (if None, will use the max reading_time)
        horizon: Number of periods to forecast
        model: The trained model to use for forecasting
    """
    try:        
        # Define features for the model
        features = [
            'minute', 'hour', 'dayofweek', 'is_weekend', 'is_surcharge_hour', 
            'time_sin', 'time_cos', 'humid', 'rain', 'temp',
            'region_Central', 'region_East', 'region_North', 'region_West',  # Note the correct order
            'taxi_lag_1', 'taxi_lag_2', 'taxi_lag_3', 'taxi_lag_6', 'taxi_lag_12',
            'taxi_rolling_4', 'taxi_rolling_12', 'taxi_rolling_24', 'taxi_rolling_48',
            'Fair', 'Fair Day', 'Fair Night', 'Fair and Warm', 'Partly Cloudy', 
            'Partly Cloudy Day', 'Partly Cloudy Night', 'Cloudy', 'Hazy', 
            'Slightly Hazy', 'Windy', 'Mist', 'Fog', 'Light Rain', 'Moderate Rain', 
            'Heavy Rain', 'Passing Showers', 'Light Showers', 'Showers', 'Heavy Showers', 
            'Thundery Showers', 'Heavy Thundery Showers', 'Heavy Thundery Showers with Gusty Winds'
        ]
        # Prepare the data
        xgb_df = _prepare_xgboost_data(df)
        
        # Generate forecasts
        forecasts = _forecast_future_with_xgboost(model, xgb_df, features, time_dim, future_periods=horizon)
        
        if forecasts.empty:
            logging.warning("No forecasts were generated, returning original dataframe")
            return df, None
        
        # Convert timestamp column to UTC
        forecasts['timestamp'] = pd.to_datetime(forecasts['timestamp'], utc=True)

        # Mapping of forecast horizons to column names in BigQuery
        horizon_mapping = {
            6: 'forecast_halfh',    # 30 minutes (6 periods)
            12: 'forecast_1h',      # 1 hour (12 periods)
            18: 'forecast_1halfh',  # 1.5 hours (18 periods)
            24: 'forecast_2h'       # 2 hours (24 periods)
        }
        
        # Find the row in the dataframe that corresponds to the execution_ts
        # This is where we'll put the forecasts
        execution_rows = df[df['reading_time'] == execution_ts]
        
        if len(execution_rows) == 0:
            logging.warning(f"No rows found for execution timestamp {execution_ts}, using most recent timestamp instead")
            execution_ts = df['reading_time'].max()
            execution_rows = df[df['reading_time'] == execution_ts]
            
            if len(execution_rows) == 0:
                logging.error("No suitable rows found for adding forecasts")
                return df
        
        # Create a copy of the original dataframe
        records_df = df.copy()
        
        # Get the regions from the execution rows
        exec_regions = execution_rows['region'].unique()
        
        # For each forecast horizon, add the forecast value to the appropriate row
        for h, col_name in horizon_mapping.items():
            # Initialize the column with NaN if it doesn't exist
            if col_name not in records_df.columns:
                records_df[col_name] = np.nan
            
            # For each region in the execution rows
            for region in exec_regions:
                # Find the forecast for this region
                region_forecasts = forecasts[forecasts['region_name'] == region]
                
                if len(region_forecasts) >= h:
                    # Get the h-step ahead forecast
                    forecast_time = region_forecasts['timestamp'].min() + pd.Timedelta(minutes=5*(h-1))
                    forecast_row = region_forecasts[region_forecasts['timestamp'] == forecast_time]
                    
                    if len(forecast_row) > 0:
                        forecast_value = forecast_row['predicted_value'].values[0]
                        
                        # Add the forecast to the execution row for this region
                        exec_region_rows = execution_rows[execution_rows['region'] == region]
                        if len(exec_region_rows) > 0:
                            for idx in exec_region_rows.index:
                                records_df.loc[idx, col_name] = forecast_value
        
        # Fill any missing forecast values with zeros
        for col_name in horizon_mapping.values():
            if col_name in records_df.columns:
                records_df[col_name] = records_df[col_name].fillna(0)
            else:
                records_df[col_name] = 0
        
        logging.info("Forecasts generated successfully")
        return records_df, forecasts
    
    except Exception as e:
        logging.error(f"Error during forecast generation: {str(e)}")
        # If forecasting fails, fill forecast columns with zeros
        for col_name in ['forecast_halfh', 'forecast_1h', 'forecast_1halfh', 'forecast_2h']:
            df[col_name] = df[col_name].fillna(0)
        return df, None