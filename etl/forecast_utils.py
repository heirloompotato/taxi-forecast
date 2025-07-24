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

def _forecast_future_with_xgboost(model, xgb_df, features, prophet_base_forecasts, execution_ts):
    """Generate forecasts for future periods by predicting residuals from Prophet with XGBoost."""
    all_forecasts = []
    
    # Predict residuals for current timestamp using XGBoost
    current_xgb_df = xgb_df[xgb_df['reading_time'] == execution_ts].copy()
    current_xgb_df['resid_2h_pred'] = model.predict(current_xgb_df[features])

    # Filter prophet_base_forecasts for values from current timestamp until 2h later
    for region in ['Central', 'North', 'East', 'West']:

        # Get the base prophet forecast for the region
        if region not in prophet_base_forecasts:
            logging.warning(f"No prophet base forecast available for region {region}, skipping")
            continue
        region_prophet_df = prophet_base_forecasts[region]
        region_prophet_df = region_prophet_df[(region_prophet_df['ds'] > execution_ts) & (region_prophet_df['ds'] <= execution_ts + pd.Timedelta(hours=2))].copy()

        # Get current region residual
        if current_xgb_df.empty:
            logging.warning(f"No current data available for region {region}, skipping forecast")
            continue
        region_xgb_df = current_xgb_df[current_xgb_df['region'] == region].copy()
        # Should only be one row, take the value
        region_xgb_residual = region_xgb_df['resid_2h_pred'].values[0]
        region_xgb_current_taxis = region_xgb_df['num_taxis'].values[0]
        
        # Create a copy of the region forecasts to avoid modifying original data
        region_forecast = region_prophet_df.copy()
        
        # Calculate number of timesteps (should be 24 for 5-minute intervals over 2 hours)
        num_timesteps = len(region_forecast)
        
        # Create weights for smooth transition from current value to residual-adjusted forecast
        # First value should be closest to current value, last value should be fully residual-adjusted
        if num_timesteps > 0:
            weights = np.linspace(0, 1, num_timesteps)
        else:
            # Handle empty dataframe case
            logging.warning(f"No timesteps found for region {region}, skipping forecast")
            continue
        
        # Calculate the target value at 2 hours (last point)
        target_value_at_2h = region_forecast.iloc[-1]['yhat'] + region_xgb_residual
        
        # Apply smoothing from current value to residual-adjusted forecast
        for i in range(len(region_forecast)):
            # Weight determines how much of the residual to apply at each step
            weight = weights[i]
            
            # Calculate adjusted forecast (gradually applying more of the residual)
            if i == len(region_forecast) - 1:
                # For the last point (2h forecast), ensure it's exactly yhat + resid_2h_pred
                region_forecast.loc[region_forecast.index[i], 'predicted_value'] = target_value_at_2h
            else:
                # For intermediate points, smoothly transition
                # Start close to current value and gradually approach the residual-adjusted forecast
                current_to_forecast_blend = region_xgb_current_taxis * (1 - weight) + region_forecast.iloc[i]['yhat'] * weight
                residual_effect = region_xgb_residual * weight
                region_forecast.loc[region_forecast.index[i], 'predicted_value'] = current_to_forecast_blend + residual_effect
            
            # Adjust upper and lower bounds by the same residual proportion
            region_forecast.loc[region_forecast.index[i], 'lower_bound_95'] = region_forecast.iloc[i]['yhat_lower'] + (region_xgb_residual * weight)
            region_forecast.loc[region_forecast.index[i], 'upper_bound_95'] = region_forecast.iloc[i]['yhat_upper'] + (region_xgb_residual * weight)
        
        # Add region name and model version
        region_forecast['region_name'] = region
        region_forecast['model_version'] = 'prophet_xgb_hybrid_v3'
        
        # Rename 'ds' to 'timestamp'
        region_forecast = region_forecast.rename(columns={'ds': 'timestamp'})
        
        # Select only the required columns
        region_forecast = region_forecast[['timestamp', 'region_name', 'predicted_value', 'lower_bound_95', 'upper_bound_95', 'model_version']]
        
        # Add to the list of all forecasts
        all_forecasts.append(region_forecast)
    
    # Combine all regional forecasts
    if all_forecasts:
        final_forecast_df = pd.concat(all_forecasts, ignore_index=True)
        return final_forecast_df
    else:
        logging.warning("No forecasts were generated")
        return pd.DataFrame(columns=['timestamp', 'region_name', 'predicted_value', 'lower_bound_95', 'upper_bound_95', 'model_version'])


def forecast_num_taxis(df, prophet_base_forecasts, model, execution_ts=None):
    """
    Generate forecasts for the next 2 hours (24 5-minute periods).
    
    Args:
        df: DataFrame with current data
        prophet_base_forecasts: DataFrame with Prophet base forecasts
        execution_ts: The execution timestamp (if None, will use the max reading_time)
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
        forecasts = _forecast_future_with_xgboost(model, xgb_df, features, prophet_base_forecasts, execution_ts)
        
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
        
        # Fill any missing forecast values with ffill if not zeros
        for col_name in horizon_mapping.values():
            if col_name in records_df.columns:
                records_df[col_name] = records_df[col_name].ffill().fillna(0)
            else:
                records_df[col_name] = 0
        
        logging.info("Forecasts generated successfully")
        return records_df, forecasts
    
    except Exception as e:
        logging.error(f"Error during forecast generation: {str(e)}")
        # If forecasting fails, ffill forecasts, if it fails fill with zeros
        for col_name in ['forecast_halfh', 'forecast_1h', 'forecast_1halfh', 'forecast_2h']:
            if col_name in df.columns:
                df[col_name] = df[col_name].ffill().fillna(0)
            else:
                df[col_name] = 0