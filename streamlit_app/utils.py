import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud import storage
from google.oauth2 import service_account
import os
import streamlit as st
import db_dtypes
import folium
import h3
from io import BytesIO
from PIL import Image

RECORDS_TABLE = os.getenv("BQ_RECORDS_TABLE")
FORECASTS_TABLE = os.getenv("BQ_FORECASTS_TABLE")
BASE_FORECASTS_TABLE = os.getenv("BQ_BASE_FORECASTS_TABLE")
MODEL_MAPE_TABLE = os.getenv("BQ_MODEL_MAPE_TABLE")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

@st.cache_data(ttl=600)
def get_current_execution_ts() -> pd.Timestamp:
    """Rounds current UTC time to the next 5-minute interval."""
    now = pd.Timestamp.utcnow()
    return now.floor('5min')

@st.cache_data()
def get_max_capacity():
    """
    Get the maximum capacity for each region, from environment variables.
    """
    return {
        'Central': int(os.getenv("MAX_CAPACITY_CENTRAL", 1187)),
        'East': int(os.getenv("MAX_CAPACITY_EAST", 948)),
        'West': int(os.getenv("MAX_CAPACITY_WEST", 441)),
        'North': int(os.getenv("MAX_CAPACITY_NORTH", 477))
    }

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

@st.cache_resource()
def get_storage_client():
    """
    Get a configured GCS Storage client.
    """
    try:
        client = storage.Client()
        print("Using Google Cloud Storage client.")
        return client
    except ImportError:
        print("Google Cloud Storage client not installed. Install with: pip install google-cloud-storage")
        return None

@st.cache_data(ttl=600)
def get_recent_records_data(cutoff, hours=6):  # Changed to 6 hours for forecast alignment
    """
    Get taxi data from BigQuery within a specific time window before the cutoff
    with a precalculated 2-hour earlier forecast for easier MAPE calculation
    
    Parameters:
    cutoff (pd.Timestamp): Upper timestamp limit (exclusive) - only records before this time will be returned
    hours (int, default=6): Hours of historical data to retrieve before the cutoff (changed to 6 for forecast alignment)

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
      WHERE reading_time <= TIMESTAMP('{cutoff_str}')
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

@st.cache_data(ttl=600)
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

@st.cache_data(ttl=600)
def get_base_forecast_data(cutoff, hours=2):
    """Get detailed forecast data from BigQuery"""
    bq_client, bq_storage_client = get_bigquery_client(use_storage_api=True)
    
    print("querying base forecast data from BigQuery...")
    
    # Ensure cutoff is a pd.Timestamp
    if not isinstance(cutoff, pd.Timestamp):
        cutoff = pd.Timestamp(cutoff)
    
    # Convert to string in BigQuery timestamp format 
    cutoff_str = cutoff.strftime('%Y-%m-%d %H:%M:%S')

    query = f"""
    SELECT ds, region, yhat
    FROM `{BASE_FORECASTS_TABLE}`
    WHERE ds > TIMESTAMP('{cutoff_str}')
        AND ds <= TIMESTAMP_ADD(TIMESTAMP('{cutoff_str}'), INTERVAL {hours} HOUR)
    ORDER BY ds, region
    """

    df = bq_client.query(query).to_dataframe(bqstorage_client=bq_storage_client)
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_convert('Singapore') # Convert to Singapore timezone

    return df

@st.cache_data()
def get_model_mape_data():
    """Get detailed model mape data from BigQuery"""
    bq_client, bq_storage_client = get_bigquery_client(use_storage_api=True)

    print("querying model mape data from BigQuery...")

    query = f"""
    SELECT *
    FROM `{MODEL_MAPE_TABLE}`
    """

    df = bq_client.query(query).to_dataframe(bqstorage_client=bq_storage_client)
    df['model_updated'] = pd.to_datetime(df['model_updated']).dt.tz_convert('Singapore') # Convert to Singapore timezone

    # Pivot model_type to columns
    df = df.pivot_table(
        index=['horizon', 'region', 'model_updated'],
        columns='model_type',
        values='mape',
        aggfunc='mean'
    ).reset_index()
    return df

@st.cache_resource()
def load_image_from_gcs(blob_path: str) -> Image.Image:
    # Initialize GCS client
    client = get_storage_client()
    print("Loading image from GCS bucket...")

    # Get the blob
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_path)

    # Download as bytes
    image_bytes = blob.download_as_bytes()

    # Load into PIL
    return Image.open(BytesIO(image_bytes))

@st.cache_resource()
def load_shap_beeswarm_from_gcs() -> dict:
    """Load SHAP beeswarm plot data from GCS"""
    # Load one shap beeswarm image for each horizon
    horizons = [0.5, 1, 1.5, 2]
    images = {}

    for horizon in horizons:
        blob_path = f"shap_beeswarm_{horizon}h.jpg"
        img = load_image_from_gcs(blob_path)
        images[horizon] = img
    
    return images

def format_sg_time(dt):
    return dt.strftime('%I.%M %p').lstrip('0').lower().replace(':', '.')

def format_sg_datetime(dt):
    date_part = dt.strftime('%d %b %Y').lstrip('0')
    time_part = dt.strftime('%I.%M %p').lstrip('0').lower().replace(':', '.')
    return f"{date_part}, {time_part}"

def create_availability_forecast(selected_timeframe=0.5):
    """
    Calculate availability metrics for each region based on selected timeframe.
    
    Args:
        selected_timeframe (float): Timeframe in hours (0.5, 1.0, 1.5, or 2.0)
    
    Returns:
        DataFrame with availability metrics for each region
    """
    # Get data from session state
    forecast_data = st.session_state.forecast_data
    base_forecast_data = st.session_state.base_forecast_data
    cutoff = st.session_state.cutoff.tz_convert('Asia/Singapore')
    
    # Calculate time window
    end_time = cutoff + timedelta(hours=selected_timeframe)
    start_time = end_time - timedelta(minutes=30)

    # Filter forecast data for the selected time window
    forecast_window = forecast_data[
        (forecast_data['timestamp'] > start_time) & 
        (forecast_data['timestamp'] <= end_time)
    ]
    
    # Filter base forecast data for the selected time window
    base_window = base_forecast_data[
        (base_forecast_data['ds'] >= start_time) & 
        (base_forecast_data['ds'] <= end_time)
    ]
    
    # Calculate metrics for each region
    regions = forecast_data['region_name'].unique()
    results = []
    
    # Calculate aggregated metrics across all regions
    # First group by region_name and get the mean for each region
    region_averages_forecast = forecast_window.groupby('region_name')['predicted_value'].mean()
    region_averages_base = base_window.groupby('region')['yhat'].mean()

    # Sum up the regional averages
    all_regions_f_taxi = region_averages_forecast.sum()
    all_regions_p_taxi = region_averages_base.sum()
    all_regions_max_cap = sum(st.session_state.max_cap.values())  # Total max capacity

    # Calculate scores for all regions combined
    all_regions_aas = min(all_regions_f_taxi / all_regions_max_cap, 1.0)
    all_regions_rel_dev = (all_regions_f_taxi - all_regions_p_taxi) / all_regions_p_taxi if all_regions_p_taxi > 0 else 0
    all_regions_ras = 1 / (1 + np.exp(-all_regions_rel_dev * 5))  # sigmoid transformation

    # Calculate composite score for all regions
    alpha = 0.5  # weight on absolute vs relative availability
    all_regions_ctas = alpha * all_regions_aas + (1 - alpha) * all_regions_ras

    # Determine recommendation and explanation for all regions
    all_regions_recommendation, all_regions_explanation = _get_recommendation_and_explanation(
        all_regions_ctas, all_regions_f_taxi, all_regions_rel_dev
    )

    # Add "All Regions" entry
    results.append({
        'region': 'All Regions',
        'f_taxi': all_regions_f_taxi,
        'p_taxi': all_regions_p_taxi,
        'rel_dev': all_regions_rel_dev,
        'aas': all_regions_aas,
        'ras': all_regions_ras,
        'ctas': all_regions_ctas,
        'recommendation': all_regions_recommendation,
        'explanation': all_regions_explanation
    })
    # Calculate metrics for individual regions
    for region in regions:
        # Get region-specific data
        region_forecast = forecast_window[forecast_window['region_name'] == region]
        region_base = base_window[base_window['region'] == region]
        
        # Calculate mean forecasted value and baseline
        f_taxi = region_forecast['predicted_value'].mean()
        p_taxi = region_base['yhat'].mean()
        
        # Get max capacity for this region
        max_cap = st.session_state.max_cap.get(region, 1000.0)  # Default if region not found
        
        # Calculate absolute availability score (AAS)
        aas = min(f_taxi / max_cap, 1.0)
        
        # Calculate relative availability score (RAS)
        rel_dev = (f_taxi - p_taxi) / p_taxi if p_taxi > 0 else 0
        ras = 1 / (1 + np.exp(-rel_dev * 5))  # sigmoid transformation
        
        # Calculate composite score (CTAS)
        alpha = 0.5  # weight on absolute vs relative availability
        ctas = alpha * aas + (1 - alpha) * ras
        
        # Determine recommendation and explanation
        recommendation, explanation = _get_recommendation_and_explanation(ctas, f_taxi, rel_dev)
        
        results.append({
            'region': region,
            'f_taxi': f_taxi,
            'p_taxi': p_taxi,
            'rel_dev': rel_dev,
            'aas': aas,
            'ras': ras,
            'ctas': ctas,
            'recommendation': recommendation,
            'explanation': explanation
        })
    
    return pd.DataFrame(results)

def _get_recommendation_and_explanation(ctas, f_taxi, rel_dev):
    """
    Generate recommendation and explanation text based on CTAS score.
    
    Args:
        ctas (float): Composite taxi availability score
        f_taxi (float): Forecasted taxi value
        rel_dev (float): Relative deviation from baseline
        
    Returns:
        tuple: (recommendation, explanation) strings
    """
    if ctas > 0.75:
        recommendation = "🟢 High availability"
        explanation = f"{f_taxi:,.0f} taxis available"
        if rel_dev > 0.1:
            explanation += f" ({rel_dev:.1%} more than expected)"
        elif rel_dev < -0.1:
            explanation += f" (still {-rel_dev:.1%} less than expected)"
    elif ctas > 0.5:
        recommendation = "🟡 Fair availability"
        explanation = f"{f_taxi:,.0f} taxis available"
        if rel_dev > 0.1:
            explanation += f" ({rel_dev:.1%} more than expected)"
        elif rel_dev < -0.1:
            explanation += f" ({-rel_dev:.1%} less than expected)"
    elif ctas > 0.3:
        recommendation = "🟠 Low availability"
        explanation = f"{f_taxi:,.0f} taxis available"
        if rel_dev > 0.1:
            explanation += f" (but {rel_dev:.1%} more than expected)"
        elif rel_dev < -0.1:
            explanation += f" ({-rel_dev:.1%} less than expected)"
    else:
        recommendation = "🔴 Very low availability"
        explanation = f"{f_taxi:,.0f} taxis available"
        if rel_dev > 0.1:
            explanation += f" (but {rel_dev:.1%} more than expected)"
        elif rel_dev < -0.1:
            explanation += f" ({-rel_dev:.1%} less than expected)"
    
    return recommendation, explanation

@st.cache_data(ttl=60, show_spinner=False)
def create_singapore_availability_map(availability_data):
    """
    Create a Singapore map with regions colored based on availability recommendations.
    
    Args:
        availability_data: DataFrame with region availability data and recommendations
    
    Returns:
        folium.Map object with colored regions
    """
    # Center coordinates for Singapore
    singapore_center = [1.3321, 103.8700]
    
    # Create a map with restricted bounds and zoom levels
    m = folium.Map(
        location=singapore_center,
        zoom_start=10,
        tiles='CartoDB positron',
        prefer_canvas=True,
        min_zoom=10,
        max_zoom=14
    )
    
    # Load H3 to region mapping if not provided
    h3_region_map = _load_h3_region_mapping()
    
    # Create a dictionary mapping region to recommendation color
    recommendation_colors = {
        "🟢": "#28a745",  # Green for "High availability"
        "🟡": "#ffc107",  # Yellow for "Fair availability"
        "🟠": "#fd7e14",  # Orange for "Low availability" 
        "🔴": "#dc3545"   # Red for "Very low availability"
    }
    
    # Create a region to color mapping from availability data
    region_color_map = {}
    
    # Default color for regions without data
    default_color = "#AAAAAA"  # Gray
    
    # Filter out "All Regions" and process individual regions
    region_data = availability_data[availability_data['region'] != 'All Regions']
    
    for _, row in region_data.iterrows():
        region = row['region']
        recommendation = row['recommendation'].split()[0]  # Get the emoji part
        region_color_map[region] = recommendation_colors.get(recommendation, default_color)
    
    # Create a feature group for individual H3 cells (with lower opacity)
    hexagon_group = folium.FeatureGroup(name="H3 Cells")
    
    # Create a new feature group specifically for region boundaries
    region_boundaries_group = folium.FeatureGroup(name="Region Boundaries")
    
    # Create a feature group for the outer boundary
    outer_boundary_group = folium.FeatureGroup(name="Outer Boundary")
    
    # Group H3 cells by region
    regions_to_h3_cells = {}
    for h3_id, region in h3_region_map.items():
        if region not in regions_to_h3_cells:
            regions_to_h3_cells[region] = []
        regions_to_h3_cells[region].append(h3_id)
    
    # Add individual H3 cells with light coloring
    for h3_id, region in h3_region_map.items():
        boundary_points = h3.cell_to_boundary(h3_id)
        polygon_coords = [[lat, lng] for lat, lng in boundary_points]
        
        color = region_color_map.get(region, default_color)
        
        # Add polygon to map with styling
        folium.Polygon(
            locations=polygon_coords,
            color="#999999",
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.4,  # Lower opacity for individual cells
            popup=None,
            tooltip=f"Region: {region}",
            name=region
        ).add_to(hexagon_group)
    
    # Add the H3 cells to the map
    hexagon_group.add_to(m)
    
    # Create region boundaries
    edge_pairs = set()
    
    # For identifying the outer boundary edges
    all_h3_ids = set(h3_region_map.keys())
    outer_edges = set()
    
    # For each H3 cell, check all its neighbors
    for h3_id, region in h3_region_map.items():
        try:
            # Get immediate neighbors (k=1)
            neighbors = h3.grid_ring(h3_id, 1)
            
            # Check each neighbor
            for neighbor in neighbors:
                # If neighbor is outside our map, this is an outer boundary edge
                if neighbor not in all_h3_ids:
                    outer_edges.add(h3_id)
                    continue
                    
                neighbor_region = h3_region_map.get(neighbor)
                
                # If neighbor exists and is from a different region, this is a boundary edge
                if neighbor_region and neighbor_region != region:
                    # Sort to ensure we don't add both (A,B) and (B,A)
                    edge = tuple(sorted([h3_id, neighbor]))
                    edge_pairs.add(edge)
        except Exception as e:
            # Skip if there's an error getting neighbors (e.g., if the cell is on the edge of the grid)
            continue
    
    # Now draw lines for each boundary edge
    for h3_id1, h3_id2 in edge_pairs:
        region1 = h3_region_map[h3_id1]
        region2 = h3_region_map[h3_id2]
        
        # Get the center points of both cells
        center1 = h3.cell_to_latlng(h3_id1)  # Returns [lat, lng]
        center2 = h3.cell_to_latlng(h3_id2)  # Returns [lat, lng]
        
        # Instead of connecting centers, we need to find the shared edge
        # Get boundaries of both cells
        boundary1 = h3.cell_to_boundary(h3_id1)  # List of [lat, lng] points
        boundary2 = h3.cell_to_boundary(h3_id2)  # List of [lat, lng] points
        
        # Find shared edge points (with some tolerance for floating point comparison)
        shared_points = []
        
        # Convert boundaries to set of tuples for easier comparison with tolerance
        boundary1_set = set((round(lat, 9), round(lng, 9)) for lat, lng in boundary1)
        boundary2_set = set((round(lat, 9), round(lng, 9)) for lat, lng in boundary2)
        
        # Find intersection
        shared_points = boundary1_set.intersection(boundary2_set)
        
        # Convert back to list format
        shared_points = [[lat, lng] for lat, lng in shared_points]
        
        # If we have exactly 2 shared points, we can draw the edge
        if len(shared_points) == 2:
            # Draw a bold line along the shared edge
            folium.PolyLine(
                locations=shared_points,
                color="black",
                weight=1.5,
                opacity=1.0,
                tooltip=f"Boundary between {region1} and {region2}"
            ).add_to(region_boundaries_group)
    
    # Add the region boundaries on top
    region_boundaries_group.add_to(m)
    
    # Draw the outer boundary of Singapore
    outer_boundary_segments = []
    
    for h3_id in outer_edges:
        # Get the boundary of this cell
        boundary_points = h3.cell_to_boundary(h3_id)
        
        # Check each edge of the hexagon
        for i in range(len(boundary_points)):
            j = (i + 1) % len(boundary_points)  # Next point (wrapping around)
            
            point1 = (round(boundary_points[i][0], 9), round(boundary_points[i][1], 9))
            point2 = (round(boundary_points[j][0], 9), round(boundary_points[j][1], 9))
            edge = tuple(sorted([point1, point2]))
            
            # Check if this edge is shared with another cell in our map
            is_outer_edge = True
            
            # Check with all immediate neighbors
            try:
                neighbors = h3.grid_ring(h3_id, 1)
                for neighbor in neighbors:
                    if neighbor not in all_h3_ids:
                        continue  # Skip neighbors outside our map
                        
                    # Check if this edge is shared with the neighbor
                    neighbor_boundary = h3.cell_to_boundary(neighbor)
                    neighbor_boundary_set = set((round(lat, 9), round(lng, 9)) for lat, lng in neighbor_boundary)
                    
                    # If both points of the edge are in the neighbor's boundary, it's a shared edge
                    if point1 in neighbor_boundary_set and point2 in neighbor_boundary_set:
                        is_outer_edge = False
                        break
            except Exception:
                pass  # Skip if error in getting neighbors
            
            if is_outer_edge:
                # This is an outer edge
                outer_boundary_segments.append([
                    [point1[0], point1[1]],
                    [point2[0], point2[1]]
                ])
    
    # Draw all outer boundary segments with a bold, black line
    for segment in outer_boundary_segments:
        folium.PolyLine(
            locations=segment,
            color="black",
            weight=1.5,
            opacity=1.0,
            tooltip="Singapore Boundary"
        ).add_to(outer_boundary_group)
    
    # Add the outer boundary on top of everything
    outer_boundary_group.add_to(m)
    
    # Define region centroids
    region_centroids = {
        'East': (1.356784118761848, 103.94642787421869), 
        'West': (1.3242262527298974, 103.70135864880591), 
        'North': (1.4143326792052733, 103.8176062436806), 
        'Central': (1.2983549939679828, 103.83328790182178)
    }
    
    # Add region labels with recommendation indicators
    for _, row in region_data.iterrows():
        region = row['region']
        recommendation = row['recommendation']
        f_taxi = row['f_taxi']
        
        # Skip if region not in our calculated centroids
        if region not in region_centroids:
            continue
        
        lat, lng = region_centroids[region]
        emoji = recommendation.split()[0]
        
        # Create popup text with detailed information
        popup_text = f"""
        <div style="min-width: 180px">
            <h4>{emoji} {region}</h4>
            <b>{' '.join(recommendation.split()[1:])}</b><br>
            <hr style="margin: 5px 0">
            <b>Forecasted Taxis:</b> {f_taxi:.1f}<br>
            <b>Relative to Baseline:</b> {row['rel_dev']:+.1%}<br>
            <b>Composite Score:</b> {row['ctas']:.2f}
        </div>
        """
        
        # Add markers with region names only (colors will be shown by the polygons)
        folium.Marker(
            location=[lat, lng],
            icon=folium.DivIcon(
                icon_size=(150, 50),
                icon_anchor=(75, 25),
                html=f'''<div style="
                    font-size: 14pt; 
                    font-weight: bold; 
                    color: black;
                    text-shadow: 1px 1px 3px white;
                    text-align: center;">
                    {emoji} {region}
                </div>'''
            ),
            popup=folium.Popup(popup_text, max_width=250)
        ).add_to(m)
    
    # Add a horizontal legend at the bottom
    legend_html = """
    <div style="
        position: fixed; 
        top: 50%; 
        right: 80px;
        transform: translateY(-50%);
        width: 110px; 
        background-color: white; 
        border: 2px solid grey; 
        z-index: 9999; 
        padding: 10px;
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.2);
        font-size: 14px;
    ">
        <div style="font-weight: bold; margin-bottom: 8px;">Availability:</div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="background-color: #28a745; width: 16px; height: 16px; margin-right: 8px; border-radius: 3px;"></span> High
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="background-color: #ffc107; width: 16px; height: 16px; margin-right: 8px; border-radius: 3px;"></span> Fair
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 6px;">
            <span style="background-color: #fd7e14; width: 16px; height: 16px; margin-right: 8px; border-radius: 3px;"></span> Low
        </div>
        <div style="display: flex; align-items: center;">
            <span style="background-color: #dc3545; width: 16px; height: 16px; margin-right: 8px; border-radius: 3px;"></span> Very Low
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

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