import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_folium import st_folium
from utils import get_recent_records_data, get_forecast_data, create_singapore_map, get_current_execution_ts


# Configure the page
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
st.markdown("Real-time taxi availability with 2-hour forecasts by region. Note that this app is in beta and may have inaccurate forecasts due to limited training data (last forecast update 24 Jul 11:30 pm). View project documentation and source code on [GitHub](https://github.com/heirloompotato/taxi-forecast).")

# Sidebar
st.sidebar.header("Controls")
selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All Regions", "Central", "North", "East", "West"]
)

hours_to_display = st.sidebar.slider(
    "Hours of Historical Data", 
    min_value=1, 
    max_value=24,
    value=4
)

# Initialize variables to avoid NameError
if 'cutoff' not in st.session_state:
    st.session_state.cutoff = get_current_execution_ts()

if 'records_data' not in st.session_state:
    st.session_state.records_data = get_recent_records_data(st.session_state.cutoff)

# Load forecast data when needed
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = get_forecast_data()

# Update filter based on current slider value
filter_time = st.session_state.cutoff - pd.Timedelta(hours=hours_to_display)

# Filter records data by time first
time_filtered_records_data = st.session_state.records_data[st.session_state.records_data['reading_time'] >= filter_time]

# Check if region selection or time filter has changed
if ("previous_region" not in st.session_state or 
        st.session_state.previous_region != selected_region or
        "previous_hours" not in st.session_state or
        st.session_state.previous_hours != hours_to_display):
    
    # Filter records data by region
    if selected_region != "All Regions":
        st.session_state.filtered_records_data = time_filtered_records_data[time_filtered_records_data['region'] == selected_region].copy()
        # Filter forecast data by region
        st.session_state.filtered_forecast_data = st.session_state.forecast_data[st.session_state.forecast_data['region_name'] == selected_region].copy()
        
        # Calculate statistics for the selected region
        st.session_state.filtered_stats_data = pd.DataFrame([{
            'min_num_taxis': st.session_state.filtered_records_data['num_taxis'].min(),
            'max_num_taxis': st.session_state.filtered_records_data['num_taxis'].max(),
            'mean_num_taxis': st.session_state.filtered_records_data['num_taxis'].mean(),
            'std_num_taxis': st.session_state.filtered_records_data['num_taxis'].std(),
            'forecast_halfh': st.session_state.filtered_records_data['forecast_halfh'].iloc[0],
            'forecast_1h': st.session_state.filtered_records_data['forecast_1h'].iloc[0],
            'forecast_1halfh': st.session_state.filtered_records_data['forecast_1halfh'].iloc[0],
            'forecast_2h': st.session_state.filtered_records_data['forecast_2h'].iloc[0],
        }])

    else:
        # For "All Regions", aggregate records data by summing across regions
        st.session_state.filtered_records_data = time_filtered_records_data.groupby('reading_time').agg({
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

        # Calculate statistics for all regions
        st.session_state.filtered_stats_data = pd.DataFrame([{
            'min_num_taxis': st.session_state.filtered_records_data['num_taxis'].min(),
            'max_num_taxis': st.session_state.filtered_records_data['num_taxis'].max(),
            'mean_num_taxis': st.session_state.filtered_records_data['num_taxis'].mean(),
            'std_num_taxis': st.session_state.filtered_records_data['num_taxis'].std(),
            'forecast_halfh': st.session_state.filtered_records_data['forecast_halfh'].iloc[0],
            'forecast_1h': st.session_state.filtered_records_data['forecast_1h'].iloc[0],
            'forecast_1halfh': st.session_state.filtered_records_data['forecast_1halfh'].iloc[0],
            'forecast_2h': st.session_state.filtered_records_data['forecast_2h'].iloc[0],
        }])

    # Update previous region and hours
    st.session_state.previous_region = selected_region
    st.session_state.previous_hours = hours_to_display

col1, col2 = st.columns([3, 7])

# COLUMN 1: MAP VIEW
with col1:
    st.subheader("Taxi Distribution Map")
    map_container = st.container()
    # Get the latest data
    try:
        latest_data = st.session_state.records_data.sort_values('reading_time').drop_duplicates(subset=['region'], keep='last')
        
        # Create the map with the taxi data
        m = create_singapore_map(latest_data)
        
        # Display the map in its dedicated container
        with map_container:
            st_folium(m, width=370, height=400)
        
    except Exception as e:
        st.error(f"Error loading map data: {str(e)}")
        st.exception(e)

    # HISTORICAL STATS
    try:
        if not st.session_state.filtered_stats_data.empty:
            st.subheader(f"Statistics for {selected_region}")
            col1a, col1b = st.columns(2)
            with col1a:
                st.metric("Avg. Taxis", f"{st.session_state.filtered_stats_data['mean_num_taxis'].values[0]:,.0f}")
                st.metric("Min. Taxis", f"{st.session_state.filtered_stats_data['min_num_taxis'].values[0]:,}")
            with col1b:
                st.metric("Max. Taxis", f"{st.session_state.filtered_stats_data['max_num_taxis'].values[0]:,}")
                st.metric("StdDev", f"{st.session_state.filtered_stats_data['std_num_taxis'].values[0]:,.0f}")
            st.subheader(f"Latest Forecasts for {selected_region}")
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("30 min forecast", f"{st.session_state.filtered_stats_data['forecast_halfh'].values[0]:,.0f}")
                st.metric("1h forecast", f"{st.session_state.filtered_stats_data['forecast_1h'].values[0]:,.0f}")
            with col2b:
                st.metric("1.5h forecast", f"{st.session_state.filtered_stats_data['forecast_1halfh'].values[0]:,.0f}")
                st.metric("2h forecast", f"{st.session_state.filtered_stats_data['forecast_2h'].values[0]:,.0f}")
    except Exception as e:
        st.warning(f"Could not load statistics: {str(e)}")


# COLUMN 2: TIME SERIES AND FORECAST
with col2:
    st.subheader("Taxi Availability & Forecast Plots")
    try:
        # Only process if we have data
        if not st.session_state.records_data.empty:
            # Create time series plot
            fig = go.Figure()
            
            if not st.session_state.filtered_records_data.empty and not st.session_state.filtered_forecast_data.empty:
                # Get the last historical data point to connect with forecast
                last_historical_point = st.session_state.filtered_records_data.iloc[-1]
                first_forecast_point = st.session_state.filtered_forecast_data.iloc[0]
                
                # Find the most recent historical data point to connect to forecast
                # Calculate time difference between last historical point and first forecast point
                # to find the closest historical point to connect with
                historical_data_sorted = st.session_state.filtered_records_data.sort_values('reading_time', ascending=False)
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
                    x=st.session_state.filtered_records_data['reading_time'], 
                    y=st.session_state.filtered_records_data['num_taxis'],
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
                    x=st.session_state.filtered_forecast_data['timestamp'], 
                    y=st.session_state.filtered_forecast_data['predicted_value'],
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
                ci_x.extend(st.session_state.filtered_forecast_data['timestamp'])
                ci_upper.extend(st.session_state.filtered_forecast_data['upper_bound_95'])
                ci_lower.extend(st.session_state.filtered_forecast_data['lower_bound_95'])
                
                # Create the confidence interval with smooth transition
                fig.add_trace(go.Scatter(
                    x=ci_x + ci_x[::-1],
                    y=ci_upper + ci_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 193, 7, 0.2)',
                    line=dict(color='rgba(255, 193, 7, 0)'),
                    name='95% CI',
                    hoverinfo='skip'  # Skip hover for the CI area
                ))
                
                # Set x-axis range to show both historical and forecast data
                x_min = min(st.session_state.filtered_records_data['reading_time'].min(), 
                          st.session_state.filtered_forecast_data['timestamp'].min())
                x_max = max(st.session_state.filtered_records_data['reading_time'].max(), 
                          st.session_state.filtered_forecast_data['timestamp'].max())
                
                # Update layout with the dynamic time range
                fig.update_layout(
                    title="Historical & Forecasted No. of Available Taxis",
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
            
            st.plotly_chart(fig, use_container_width=True)
            # Second graph - Historical data with 2-hour forecast comparison
            
            fig2 = go.Figure()

            if not st.session_state.filtered_records_data.empty:
                # Add historical actual data
                fig2.add_trace(go.Scatter(
                    x=st.session_state.filtered_records_data['reading_time'], 
                    y=st.session_state.filtered_records_data['num_taxis'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#1E88E5', width=2),
                    marker=dict(size=5, color='#1E88E5'),
                    hovertemplate='Historical:<br><b>%{y:,.0f}</b><extra></extra>'
                ))
                
                # Add the 2-hour earlier forecast line (already filtered in SQL)
                fig2.add_trace(go.Scatter(
                    x=st.session_state.filtered_records_data['reading_time'], 
                    y=st.session_state.filtered_records_data['forecast_2h_earlier'],
                    mode='lines',
                    name='2-Hour Forecast',
                    line=dict(color='#FFC107', width=2),  # Same color as forecast but solid line
                    hovertemplate='2-Hour Forecast:<br><b>%{y:,.0f}</b><extra></extra>'

                ))
                
                # Calculate MAPE
                mask = st.session_state.filtered_records_data['num_taxis'] != 0
                st.session_state.filtered_records_data.loc[mask, 'abs_percent_error'] = abs(
                    (st.session_state.filtered_records_data.loc[mask, 'num_taxis'] - st.session_state.filtered_records_data.loc[mask, 'forecast_2h_earlier']) /
                    st.session_state.filtered_records_data.loc[mask, 'num_taxis']
                ) * 100
                mape = st.session_state.filtered_records_data['abs_percent_error'].mean()
                
                # Add MAPE as an annotation
                fig2.add_annotation(
                    xref="paper", yref="paper",
                    x=0.01, y=0.1,
                    text=f"2-Hour Forecast MAPE: {mape:,.2f}%",
                    showarrow=False,
                    font=dict(size=14),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#FFC107",
                    borderwidth=2,
                    borderpad=4,
                    align="left"
                )
                
                # Set x-axis range to show data
                x_min = st.session_state.filtered_records_data['reading_time'].min()
                x_max = st.session_state.filtered_records_data['reading_time'].max()
                
                # Update layout with the dynamic time range
                fig2.update_layout(
                    title="Historical vs 2-Hour Forecast Comparison",
                    xaxis=dict(
                        range=[x_min, x_max],
                        title='Time'
                    ),
                    yaxis_tickformat = ",",
                    yaxis_title='Number of Taxis',
                    margin=dict(l=20, r=20, t=50, b=20),
                    height=500,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    hovermode="x unified"
                )

            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.warning("No taxi data available. Please check your data source.")
    
    except Exception as e:
        st.error(f"Error loading time series data: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.caption("Data updates every 5 minutes | Last updated: " + 
        st.session_state.cutoff.tz_convert('Asia/Singapore').strftime("%Y-%m-%d %H:%M:%S"))

with st.expander("ℹ️ About this App"):
    st.write("""This app provides accurate real-time taxi availability data for Singapore using open source LTA APIs and forecasts with data from open source NEA weather APIs. The forecasting employs a two-stage hybrid approach: First, Prophet generates baseline forecasts using historical taxi demand patterns. Then, XGBoost refines these forecasts by predicting the residuals 2 hours ahead using additional features including rolling averages, lagged values, weather conditions, and temporal indicators. This hybrid model captures both the long-term seasonality through Prophet and short-term fluctuations through XGBoost, delivering more accurate taxi demand predictions.""")