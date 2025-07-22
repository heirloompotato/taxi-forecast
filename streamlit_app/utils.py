import pandas as pd
from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.oauth2 import service_account
import os
import streamlit as st
import db_dtypes
import folium
import h3

RECORDS_TABLE = os.getenv("BQ_RECORDS_TABLE")
FORECASTS_TABLE = os.getenv("BQ_FORECASTS_TABLE")

@st.cache_data(ttl=300)
def get_current_execution_ts() -> pd.Timestamp:
    """Rounds current UTC time to the next 5-minute interval."""
    now = pd.Timestamp.utcnow()
    return now.floor('5min')

@st.cache_resource
def get_bigquery_client(use_storage_api=True):
    """
    Get a configured BigQuery client with appropriate credentials.
    Handles both local development and cloud deployment scenarios.
    """
    # For GCP Cloud Run, use the default service account credentials
    client = bigquery.Client()
    print(f"Using GCP default credentials")
    
    if use_storage_api:
        try:
            storage_client = bigquery_storage.BigQueryReadClient()
            print("Using BigQuery Storage API for faster data transfers.")
            return client, storage_client
        except ImportError:
            print("BigQuery Storage API not installed. For better performance, install with: pip install google-cloud-bigquery[storage]")
            return client
    
    return client

@st.cache_data(ttl=300)
def get_recent_records_data(cutoff, hours=26):  # Changed to 26 hours to get data for forecast alignment
    """
    Get taxi data from BigQuery within a specific time window before the cutoff
    with a precalculated 2-hour earlier forecast for easier MAPE calculation
    
    Parameters:
    cutoff (pd.Timestamp): Upper timestamp limit (exclusive) - only records before this time will be returned
    hours (int, default=26): Hours of historical data to retrieve before the cutoff (extended to 26 for forecast alignment)
    
    Returns:
    DataFrame: Records data filtered by time window with aligned forecast columns
    """
    bq_client, bq_storage_client = get_bigquery_client(use_storage_api=True)
    print("querying recent taxi data from BigQuery...")
    
    # Ensure cutoff is a pd.Timestamp
    if not isinstance(cutoff, pd.Timestamp):
        cutoff = pd.Timestamp(cutoff)
    
    # Convert to string in BigQuery timestamp format 
    cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')
    
    # Query data from the time window: (cutoff - hours) to cutoff
    # Include a self-join to get the forecast_2h from 2 hours earlier
    query = f"""
    WITH base_data AS (
      SELECT reading_time, region, num_taxis, 
             forecast_halfh, forecast_1h, forecast_1halfh, forecast_2h
      FROM `{RECORDS_TABLE}`
      WHERE reading_time < TIMESTAMP('{cutoff_str}')
        AND reading_time >= TIMESTAMP_SUB(TIMESTAMP('{cutoff_str}'), INTERVAL {hours} HOUR)
    )
    
    SELECT 
      curr.reading_time,
      curr.region,
      curr.num_taxis,
      curr.forecast_halfh,
      curr.forecast_1h, 
      curr.forecast_1halfh,
      curr.forecast_2h,
      -- Get the 2h forecast made 2 hours earlier (if available)
      prev.forecast_2h AS forecast_2h_earlier
    FROM 
      base_data AS curr
    LEFT JOIN
      base_data AS prev
    ON
      curr.region = prev.region
      -- Join with data from 2 hours earlier
      AND curr.reading_time = TIMESTAMP_ADD(prev.reading_time, INTERVAL 2 HOUR)
    WHERE prev.forecast_2h IS NOT NULL
    ORDER BY curr.reading_time DESC, curr.region
    """
    df = bq_client.query(query).to_dataframe(bqstorage_client=bq_storage_client)
    df['reading_time'] = pd.to_datetime(df['reading_time']).dt.tz_convert('Singapore') # Convert to Singapore timezone

    return df 

@st.cache_data(ttl=300)
def get_forecast_data():
    """Get detailed forecast data from BigQuery"""
    bq_client, bq_storage_client = get_bigquery_client(use_storage_api=True)
    
    print("querying forecast data from BigQuery...")
    
    query = f"""
    SELECT timestamp, region_name, predicted_value, 
           lower_bound_95, upper_bound_95
    FROM `{FORECASTS_TABLE}`
    ORDER BY timestamp, region_name
    """

    df = bq_client.query(query).to_dataframe(bqstorage_client=bq_storage_client)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert('Singapore') # Convert to Singapore timezone

    return df

def _load_h3_region_mapping():
    mapping = {
        "876526ac8ffffff": "East",
        "876520ca8ffffff": "West",
        "876526368ffffff": "North",
        "876526348ffffff": "North",
        "876520d89ffffff": "West",
        "876526ac9ffffff": "Central",
        "876520ca9ffffff": "West",
        "876526369ffffff": "North",
        "876526349ffffff": "North",
        "876520d8affffff": "West",
        "876526acaffffff": "East",
        "876520caaffffff": "West",
        "876520c86ffffff": "West",
        "87652636affffff": "North",
        "876520d8bffffff": "West",
        "876526acbffffff": "East",
        "876520ca6ffffff": "West",
        "876520cabffffff": "West",
        "87652636bffffff": "North",
        "876520c0cffffff": "West",
        "876526accffffff": "East",
        "876520cacffffff": "West",
        "87652636cffffff": "North",
        "876520c8cffffff": "West",
        "87652634cffffff": "North",
        "876526acdffffff": "Central",
        "876520cadffffff": "West",
        "87652636dffffff": "North",
        "87652634dffffff": "North",
        "876520c0effffff": "West",
        "876520d8effffff": "Central",
        "876526aceffffff": "East",
        "876520caeffffff": "West",
        "87652636effffff": "North",
        "876520c8effffff": "West",
        "87652634effffff": "North",
        "876520d90ffffff": "Central",
        "876526ad0ffffff": "East",
        "876520cb0ffffff": "West",
        "876520c94ffffff": "North",
        "876520c90ffffff": "North",
        "876520db1ffffff": "Central",
        "876520c11ffffff": "West",
        "876520d91ffffff": "Central",
        "876526ad1ffffff": "East",
        "876520cb1ffffff": "West",
        "876526371ffffff": "East",
        "876520c91ffffff": "North",
        "876520d86ffffff": "Central",
        "876520db2ffffff": "Central",
        "876520c12ffffff": "West",
        "876520d92ffffff": "Central",
        "876520cb2ffffff": "North",
        "876520db3ffffff": "Central",
        "876520c13ffffff": "West",
        "876526af3ffffff": "East",
        "876520d93ffffff": "Central",
        "876526ad3ffffff": "East",
        "876520cb3ffffff": "North",
        "876520c14ffffff": "West",
        "876520d94ffffff": "Central",
        "876526ad4ffffff": "East",
        "876520cb4ffffff": "West",
        "87652634affffff": "North",
        "876520c15ffffff": "West",
        "876520d95ffffff": "Central",
        "876526ad5ffffff": "East",
        "876520cb5ffffff": "West",
        "876526375ffffff": "East",
        "876520c95ffffff": "North",
        "876520c16ffffff": "West",
        "876520d96ffffff": "Central",
        "876520cb6ffffff": "North",
        "876520c18ffffff": "West",
        "876520d98ffffff": "Central",
        "876526ad8ffffff": "East",
        "876520d99ffffff": "West",
        "876526ad9ffffff": "East",
        "87652634bffffff": "North",
        "876520c1affffff": "West",
        "876520d9affffff": "North",
        "876526345ffffff": "North",
        "876520d9bffffff": "West",
        "876526adbffffff": "East",
        "876520c1cffffff": "West",
        "876520d9cffffff": "Central",
        "876526ac6ffffff": "East",
        "876526adcffffff": "East",
        "876520c9cffffff": "West",
        "876520c1dffffff": "West",
        "876520d9dffffff": "Central",
        "876526addffffff": "East",
        "876520c9dffffff": "West",
        "87652635dffffff": "North",
        "876520c1effffff": "West",
        "876520d9effffff": "Central",
        "876526adeffffff": "East",
        "876520d80ffffff": "Central",
        "876520d88ffffff": "West",
        "876520c10ffffff": "West",
        "876520c06ffffff": "West",
        "876526ac0ffffff": "East",
        "876520ca0ffffff": "West",
        "876526360ffffff": "East",
        "876520c80ffffff": "West",
        "876526340ffffff": "North",
        "876520d81ffffff": "Central",
        "876526ac1ffffff": "East",
        "876520ca1ffffff": "West",
        "876526361ffffff": "North",
        "876520c81ffffff": "West",
        "876526366ffffff": "East",
        "876526341ffffff": "North",
        "876520da2ffffff": "Central",
        "876520c02ffffff": "West",
        "876520d82ffffff": "Central",
        "876526ac2ffffff": "East",
        "876520ca2ffffff": "West",
        "876526362ffffff": "East",
        "876520c82ffffff": "West",
        "876520da3ffffff": "Central",
        "876520c03ffffff": "West",
        "876520d83ffffff": "Central",
        "876526ac3ffffff": "East",
        "876520ca3ffffff": "West",
        "876526363ffffff": "North",
        "876520c83ffffff": "West",
        "876526343ffffff": "North",
        "876520d84ffffff": "Central",
        "876520ca4ffffff": "West",
        "876526364ffffff": "East",
        "876520c84ffffff": "West",
        "876526344ffffff": "North",
        "876520c01ffffff": "West",
        "876520c00ffffff": "West",
        "876520c05ffffff": "West",
        "876520d85ffffff": "Central",
        "876520ca5ffffff": "West",
        "876526365ffffff": "East",
        "876520c85ffffff": "West"
    }
    return mapping

# Function to create the map with H3 cells
@st.cache_data(ttl=300, show_spinner=False)
def create_singapore_map(latest_data):
    # Center coordinates for Singapore - adjusted slightly to the right to compensate for the shift
    singapore_center = [1.3321, 103.8700]  # Shifted slightly east to compensate
    
    # Create a simple gray background map with reduced zoom
    m = folium.Map(
        location=singapore_center,
        zoom_start=10,  # Less zoomed in
        tiles='CartoDB positron',  # Light gray map
        prefer_canvas=True
    )
    
    # Load H3 to region mapping
    h3_region_map = _load_h3_region_mapping()
    
    # Define colors for each region
    region_colors = {
        "Central": "#FF5733",  # Red-orange
        "North": "#33A8FF",    # Blue
        "East": "#33FF57",     # Green
        "West": "#D133FF"      # Purple
    }
    
    # Create a feature group for hexagons
    hexagon_group = folium.FeatureGroup(name="H3 Cells")
    
    # Track regions for centroids calculation
    region_hexagons = {}
    
    # First pass: collect all hexagons by region
    for h3_id, region in h3_region_map.items():
        boundary_points = h3.cell_to_boundary(h3_id)
        polygon_coords = [[lat, lng] for lat, lng in boundary_points]
        
        if region not in region_hexagons:
            region_hexagons[region] = []
        region_hexagons[region].append(polygon_coords)
        
        color = region_colors.get(region, "#AAAAAA")
        
        # Add polygon to map with styling and popup
        folium.Polygon(
            locations=polygon_coords,
            color="#333333",
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.3,
            popup=f"Region: {region}",
            tooltip=f"Region: {region}",
            name=region
        ).add_to(hexagon_group)

    # Add hexagons to map
    hexagon_group.add_to(m)
    

    region_centroids = {
        'East': (1.356784118761848, 103.94642787421869), 
        'West': (1.3242262527298974, 103.70135864880591), 
        'North': (1.4143326792052733, 103.8176062436806), 
        'Central': (1.2983549939679828, 103.83328790182178)}
    
    # Add taxi data to map
    if latest_data is not None and not latest_data.empty:
        for _, row in latest_data.iterrows():
            region = row['region']
            num_taxis = row['num_taxis']
            forecast_2h = row.get('forecast_2h', 'N/A')  # Use get in case forecast is not available
            
            # Skip if region not in our calculated centroids
            if region not in region_centroids:
                continue
                
            lat, lng = region_centroids[region]
            
            # Create popup text for when users click
            popup_text = f"""
            <b>{region}</b><br>
            Current Taxis: {num_taxis}<br>
            """
            
            if forecast_2h != 'N/A':
                popup_text += f"2h Forecast: {forecast_2h:.1f}<br>"
            
            # Simple black bolded font as requested
            folium.Marker(
                location=[lat, lng],
                icon=folium.DivIcon(
                    icon_size=(150, 50),
                    icon_anchor=(75, 25),
                    html=f'''<div style="
                        font-size: 14pt; 
                        font-weight: bold; 
                        color: black;
                        text-align: center;">
                        {region}<br>{num_taxis}
                    </div>'''
                ),
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(m)
    
    return m