import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def forecast_num_taxis(df: pd.DataFrame) -> pd.DataFrame:
    logging.info(f"Starting forecast for {df.shape[0]} records.")
    forecast_cols = ['forecast_halfh', 'forecast_1h', 'forecast_1halfh', 'forecast_2h']

    # Temporarily fill all forecast values with 0
    for col in forecast_cols:
        df[col] = df[col].fillna(0)
    return df