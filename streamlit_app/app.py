import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils import get_max_capacity,get_recent_records_data, get_forecast_data, get_base_forecast_data, get_model_mape_data, create_singapore_availability_map, get_current_execution_ts, create_availability_forecast, load_image_from_gcs, load_shap_beeswarm_from_gcs, format_sg_time, format_sg_datetime
from streamlit_folium import st_folium

def refresh_data():
    # Update last refresh time
    st.session_state.last_refresh_time = pd.Timestamp.utcnow().tz_convert('Asia/Singapore')
    # Clear initialized flag to force reload
    st.session_state.data_initialized = False
    st.session_state.refresh_requested = True

# Function to initialize or refresh data
def initialize_or_refresh_data():
    """Initialize all data on first load or selectively refresh data when requested"""
    # Determine if this is initial load or a refresh
    is_initial_load = "data_initialized" not in st.session_state
    is_refresh = not is_initial_load and st.session_state.data_initialized == False

    if is_initial_load or is_refresh:
        print("Refreshing or initializing data")
        with st.spinner("Loading data..." if is_initial_load else "Refreshing data..."):
            data = {}
            
            # Always refresh these items
            data["cutoff"] = get_current_execution_ts()
            print(f"Cutoff time: {data['cutoff']}")
            data["last_refresh_time"] = st.session_state.get("last_refresh_time", pd.Timestamp.utcnow().tz_convert('Asia/Singapore'))
            data["records_data"] = get_recent_records_data(data["cutoff"], hours=6)
            data["forecast_data"] = get_forecast_data()
            data["base_forecast_data"] = get_base_forecast_data(data["cutoff"])
            
            # Only load static data during initial load
            if is_initial_load:
                data["model_mape_data"] = get_model_mape_data()
                data["shap_beeswarm_dict"] = load_shap_beeswarm_from_gcs()
                data["arc_diagram"] = load_image_from_gcs("arc_diagram.jpg")
                data["max_cap"] = get_max_capacity()
            
            # Update session state with dynamic data
            for key, value in data.items():
                st.session_state[key] = value
            
            # Mark data as initialized and clear refresh flag
            st.session_state.data_initialized = True
            if 'refresh_requested' in st.session_state:
                del st.session_state.refresh_requested

def render_taxi_availability_section():
    """Main function to render the taxi availability section"""
    st.subheader("Taxi Availability Forecast")
    
    # Check if required session state variables exist
    required_vars = ['forecast_data', 'base_forecast_data', 'cutoff']
    missing_vars = [var for var in required_vars if var not in st.session_state]
    
    if missing_vars:
        st.error(f"Missing required data: {', '.join(missing_vars)}")
        return
    
    # Use segmented_control for timeframe selection
    timeframe_options = {
        "30 min": 0.5, 
        "1 hour": 1.0, 
        "1.5 hours": 1.5, 
        "2 hours": 2.0
    }
        
    selected_option = st.segmented_control(
        "Select timeframe",
        options=list(timeframe_options.keys()),
        default="30 min",  # Default value
        key="availability_timeframe_selection"  # Use a unique key to avoid conflicts
    )

    # Simply use the selected value directly
    st.session_state.selected_timeframe = timeframe_options[selected_option]
    
    # Get availability data based on selected timeframe
    availability_data = create_availability_forecast(st.session_state.selected_timeframe)

    cutoff = st.session_state.cutoff.tz_convert('Asia/Singapore')
    end_time = cutoff + timedelta(hours=st.session_state.selected_timeframe)
    start_time = end_time - timedelta(minutes=25)

    st.caption(f"Showing forecast from {format_sg_time(start_time)} to {format_sg_time(end_time)}")

    # Extract the "All Regions" row
    all_regions_data = availability_data[availability_data['region'] == 'All Regions'].iloc[0]
    
    # Display the "All Regions" recommendation with a divider and larger font
    st.markdown("### Overall Forecast")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(
            f"<span style='font-size:20px; font-weight:bold'>{all_regions_data['recommendation'].split()[0]} All Regions</span>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(f"**{' '.join(all_regions_data['recommendation'].split()[1:])}** - {all_regions_data['explanation']}")

    # Display region-specific recommendations
    st.markdown("### Region Forecasts")

    # Make sure the map has a fixed key that doesn't change between sessions
    if 'map_instance_id' not in st.session_state:
        st.session_state.map_instance_id = 0

    # Ensure map rendering state is initialized
    if 'map_rendered' not in st.session_state:
        st.session_state.map_rendered = False
    
    map_container = st.container()
    with map_container:
        # Create map using the updated function
        availability_map = create_singapore_availability_map(availability_data)
        # Use st_folium with the right parameters
        folium_output = st_folium(
            availability_map, 
            width=600, 
            height=250,
            key=f"folium_map_{st.session_state.get('map_instance_id', 0)}",
            returned_objects=[],
            feature_group_to_add=None,
            # These settings help prevent rerenders
            center=None,
            zoom=None
        )
        # Force rerender after initial load, to ensure map is displayed correctly
        if not st.session_state.map_rendered:
            st.session_state.map_rendered = True
            st.rerun()
    
    # Filter out "All Regions" and sort remaining regions by Central, West, North, East
    region_data = availability_data[availability_data['region'] != 'All Regions'].copy()
    region_order = ['Central', 'West', 'North', 'East']
    region_data['region'] = pd.Categorical(region_data['region'], categories=region_order, ordered=True)
    region_data = region_data.sort_values('region')

    for _, row in region_data.iterrows():
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(
                f"<span style='font-size:20px; font-weight:bold'>{row['recommendation'].split()[0]} {row['region']}</span>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(f"**{' '.join(row['recommendation'].split()[1:])}** - {row['explanation']}")
    
    # Show detailed metrics in expander
    with st.expander("Show detailed metrics"):
        # Create a more readable metrics table
        metrics_df = availability_data[['region', 'f_taxi', 'p_taxi', 'rel_dev', 'aas', 'ras', 'ctas']]
        metrics_df = metrics_df.rename(columns={
            'region': 'Region',
            'f_taxi': 'Forecasted Taxis',
            'p_taxi': 'Baseline Taxis',
            'rel_dev': 'Relative Deviation (%)',
            'aas': 'Absolute Score',
            'ras': 'Relative Score',
            'ctas': 'Composite Score'
        })
        
        # Format numeric columns
        format_dict = {
            'Forecasted Taxis': '{:.1f}',
            'Baseline Taxis': '{:.1f}',
            'Relative Deviation (%)': '{:+.1%}',
            'Absolute Score': '{:.2f}',
            'Relative Score': '{:.2f}',
            'Composite Score': '{:.2f}'
        }
        
        for col, fmt in format_dict.items():
            metrics_df[col] = metrics_df[col].apply(lambda x: fmt.format(x))

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Add footnotes to explain metrics
        st.markdown("### Metrics Explained")
        st.markdown("""
        **Forecasted Taxis**: Number of taxis predicted to be available based on our XGBoost+Prophet hybrid model.
        
        **Baseline Taxis**: Expected number of taxis based on historical patterns for this specific time and day from Prophet model.

        **Relative Deviation**: Percentage difference between forecasted taxis and baseline (positive = more taxis than typical).

        **Absolute Score**: Ratio of forecasted taxis to maximum capacity (0-1 scale), indicating absolute availability.

        **Relative Score**: Transformed measure of how current forecast compares to typical baseline (0-1 scale).

        **Composite Score**: Combined score that balances absolute availability (75% weightage) with relative comparison (25% weightage) to determine recommendations.
        """)
        
        st.caption("* Maximum capacity values are based on 99th percentile of historical observations for each region.")

def plot_historical_w_forecast_taxis():
    """Plot historical data and forecasted values"""
    st.subheader("Historical Data with Forecast")
        # Create the segmented control

    region_options = ["All Regions", "Central", "East", "West", "North"]

    selected_region = st.segmented_control(
        "Select region",
        options=region_options,
        default="All Regions",  # Default value
        key="region_selection"  # Use a unique key to avoid conflicts
    )

    # Simply use the selected value directly
    st.session_state.selected_region = selected_region
    
    # Filter records data by region
    if selected_region != "All Regions":
        st.session_state.filtered_records_data = st.session_state.records_data[st.session_state.records_data['region'] == selected_region].copy()
        # Filter forecast data by region
        st.session_state.filtered_forecast_data = st.session_state.forecast_data[st.session_state.forecast_data['region_name'] == selected_region].copy()
    else:
        # For "All Regions", aggregate records data by summing across regions
        st.session_state.filtered_records_data = st.session_state.records_data.groupby('reading_time').agg({
            'num_taxis': 'sum',
            'forecast_halfh': 'sum',
            'forecast_1h': 'sum',
            'forecast_1halfh': 'sum',
            'forecast_2h': 'sum',
            'forecast_2h_earlier': 'sum'
        }).reset_index()
        
        # Aggregate forecast data for all regions
        st.session_state.filtered_forecast_data = st.session_state.forecast_data.groupby('timestamp').agg({
            'predicted_value': 'sum',
            'lower_bound_95': 'sum',
            'upper_bound_95': 'sum'
        }).reset_index()

    fig = _generate_historical_w_forecast_taxis_plot(
        st.session_state.filtered_records_data, 
        st.session_state.filtered_forecast_data
    )

    st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=60)
def _generate_historical_w_forecast_taxis_plot(filtered_records_data, filtered_forecast_data):
    """Generate the plot for historical data with forecasted values"""
    fig = go.Figure()
    # Get the last historical data point to connect with forecast
    first_forecast_point = filtered_forecast_data.iloc[0]
    
    # Find the most recent historical data point to connect to forecast
    # Calculate time difference between last historical point and first forecast point
    # to find the closest historical point to connect with
    historical_data_sorted = filtered_records_data.sort_values('reading_time', ascending=False)
    first_forecast_time = first_forecast_point['timestamp']
    
    # Find the closest historical point to the first forecast point
    closest_historical = historical_data_sorted.iloc[0]
    min_time_diff = abs((closest_historical['reading_time'] - first_forecast_time).total_seconds())
    
    for _, row in historical_data_sorted.iterrows():
        time_diff = abs((row['reading_time'] - first_forecast_time).total_seconds())
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_historical = row
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=filtered_records_data['reading_time'], 
        y=filtered_records_data['num_taxis'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1E88E5', width=2),
        marker=dict(size=5, color='#1E88E5'),
        hovertemplate='Historical:<br><b>%{y:,.0f}</b><extra></extra>'
    ))
    
    # Add a connector line from historical to forecast (invisible in hover)
    fig.add_trace(go.Scatter(
        x=[closest_historical['reading_time'], first_forecast_point['timestamp']],
        y=[closest_historical['num_taxis'], first_forecast_point['predicted_value']],
        mode='lines',
        line=dict(color='#FFC107', width=2, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add the forecast line (starting from the second point to avoid duplicate)
    fig.add_trace(go.Scatter(
        x=filtered_forecast_data['timestamp'], 
        y=filtered_forecast_data['predicted_value'],
        mode='lines',
        name='Forecast',
        line=dict(color='#FFC107', width=2, dash='dash'),
        hovertemplate='Forecast:<br><b>%{y:,.0f}</b><extra></extra>'
    ))
    
    # Create a smooth transition for the confidence interval
    # Start the CI from the last historical point
    ci_x = [closest_historical['reading_time']]
    ci_upper = [closest_historical['num_taxis']]  # Start at the historical value
    ci_lower = [closest_historical['num_taxis']]  # Start at the historical value
    
    # Add all forecast CI points
    ci_x.extend(filtered_forecast_data['timestamp'])
    ci_upper.extend(filtered_forecast_data['upper_bound_95'])
    ci_lower.extend(filtered_forecast_data['lower_bound_95'])
    
    # Create the confidence interval with smooth transition
    fig.add_trace(go.Scatter(
        x=ci_x + ci_x[::-1],
        y=ci_upper + ci_lower[::-1],
        fill='toself',
        fillcolor='rgba(255, 193, 7, 0.2)',
        line=dict(color='rgba(255, 193, 7, 0)'),
        name='90% CI',
        hoverinfo='skip'  # Skip hover for the CI area
    ))
    
    # Set x-axis range to show both historical and forecast data
    x_min = min(filtered_records_data['reading_time'].min(), 
            filtered_forecast_data['timestamp'].min())
    x_max = max(filtered_records_data['reading_time'].max(), 
            filtered_forecast_data['timestamp'].max())
    
    # Update layout with the dynamic time range
    fig.update_layout(
        xaxis=dict(
            range=[x_min, x_max],
            title='Time'
        ),
        yaxis_tickformat = ",",
        yaxis_title='Number of Taxis',
        margin=dict(l=20, r=20, t=30, b=20),
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode="x unified"
    )
    return fig

def render_model_performance():
    """Show model perforamnce and SHAP feature importance"""
    st.subheader("Model Performance")

    model_timeframe_options = {
        "30 min": 0.5, 
        "1 hour": 1.0, 
        "1.5 hours": 1.5, 
        "2 hours": 2.0
    }

    selected_model_option = st.segmented_control(
        "Select timeframe",
        options=list(model_timeframe_options.keys()),
        default="30 min",
        key="model_timeframe_selection" # Use a unique key to avoid conflicts
    )

    # Simply use the selected value directly
    st.session_state.selected_model_timeframe = model_timeframe_options[selected_model_option]

    # Fetch and display model MAPE data
    if "model_mape_data" in st.session_state:
        # Filter data by horizon
        model_mape_data = st.session_state.model_mape_data[st.session_state.model_mape_data['horizon'] == st.session_state.selected_model_timeframe]
        if not model_mape_data.empty:
            # Get model_updated timestamp and describe model)
            model_updated_time = format_sg_datetime(model_mape_data['model_updated'].iloc[0])
            st.markdown(f"""
            **Model in Production**: Ensemble model using Prophet + XGBoost trained on Prophet residuals
            
            **Model updated at:** {model_updated_time} (Singapore Time)
            """)
            # Create a table with MAPE values
            model_mape_table = model_mape_data[['region', 'Prophet', 'Ensemble']].copy()
            model_mape_table = model_mape_table.rename(columns={
            'region': 'Region',
            'Prophet': 'Prophet MAPE (%)',
            'Ensemble': 'Ensemble MAPE (%)'
            })
            model_format_dict = {'Prophet MAPE (%)': '{:.1%}',
                           'Ensemble MAPE (%)': '{:.1%}'}
            for col, fmt in model_format_dict.items():
                model_mape_table[col] = model_mape_table[col].apply(lambda x: fmt.format(x))

            st.caption(f"Model MAPE for {selected_model_option}")
            st.dataframe(model_mape_table, use_container_width=True, hide_index=True)

    st.subheader(f"XGBoost Feature Importance")
    # Fetch and display image
    if "shap_beeswarm_dict" in st.session_state:
        shap_beeswarm = st.session_state.shap_beeswarm_dict.get(st.session_state.selected_model_timeframe)
        if shap_beeswarm:
            st.caption(f"SHAP Beeswarm for {selected_model_option}")
            st.image(shap_beeswarm, use_container_width=True)
        else:
            st.info("No SHAP plot available for the selected timeframe.")
    else:
        st.warning("SHAP plots have not been loaded.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Taxi Availability Forecast",
        page_icon="🚕",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Add custom CSS for styling
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
        }
        .st-emotion-cache-1kyxreq {
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🚕 Singapore Taxi Availability Forecast [BETA]")
    st.markdown("Real-time taxi availability with 2-hour forecasts by region. Note that this app is in beta and may have inaccurate forecasts due to limited training data. View project documentation and source code on [GitHub](https://github.com/heirloompotato/taxi-forecast).")

    # Calculate time since last refresh
    current_time = pd.Timestamp.utcnow().tz_convert('Asia/Singapore')
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = current_time
    
    time_since_refresh = current_time - st.session_state.last_refresh_time
    minutes_since_refresh = time_since_refresh.total_seconds() / 60
    
    # Create refresh button with conditional enabling
    refresh_col1, refresh_col2 = st.columns([3, 7])
    with refresh_col1:
        if minutes_since_refresh >= 5:
            # Use on_click parameter to trigger the callback function
            st.button('🔄 Refresh All Data', on_click=refresh_data, key="refresh_button")
            minutes_remaining = 0
        else:
            # Disabled button with countdown
            minutes_remaining = max(0, 5 - minutes_since_refresh)
            st.button('🔄 Refresh All Data', disabled=True, key="refresh_button_disabled")

    with refresh_col2:
        # Show when data was last updated
        refresh_text = f"""<div style='font-size: 0.85rem; color: gray;'>
            Data last updated: {format_sg_time(st.session_state.last_refresh_time)}<br>"""
        
        if minutes_since_refresh >= 5:
            refresh_text += "Refresh available now</div>"
        else:
            refresh_text += f"Next refresh available in {min(int(minutes_remaining) + 1,5)} minute(s)</div>"
        
        st.markdown(refresh_text, unsafe_allow_html=True)

    # Initialize or refresh data where required
    initialize_or_refresh_data()

    col1, col2 = st.columns([3, 2])
    with col1:
        render_taxi_availability_section()
        st.divider()
        plot_historical_w_forecast_taxis()
    with col2:
        st.subheader("Architecture Diagram")
        # Use the cached image from session state
        st.image(st.session_state.arc_diagram, use_container_width=True)
        st.markdown("""
            This diagram illustrates the architecture of the taxi availability forecasting system, including data sources, processing pipelines, and model components.
        """)
        st.divider()
        performance_container = st.container()
        with performance_container:
            render_model_performance()