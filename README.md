# Singapore Taxi Availability Forecast [BETA]

This project is a real-time forecasting tool that estimates **2-hour future taxi availability across Singapore** using live location data and weather conditions. Built with a full GCP-based pipeline and visualized through an interactive Streamlit app, this tool aims to provide **insights into short-term mobility trends**. <br>
🟢 Try the live beta app at [sgtaxiforecast.com](https://sgtaxiforecast.com)

## 🚕 Problem Statement & Business Relevance

**Taxi availability** acts as a useful proxy for urban transport **demand and supply dynamics**. Understanding how availability shifts based on time, weather, and location can help:

- **Consumers**: Anticipate peak periods and plan ahead when booking a taxi or ride-hailing service.
- **Transport Operators**: Optimize driver deployment and reduce passenger wait times.
- **Urban Planners & Government Agencies**: Gain insights into commuter behavior and demand hotspots to inform mobility policy or infrastructure planning.
- **Logistics & Delivery Firms**: Strategize dispatching by aligning with short-term traffic and transport flow.

This tool showcases how **open transport data** can be combined with **cloud infrastructure** and **machine learning** to produce actionable insights in a scalable and reproducible manner.

## 📁 Project Structure
├── collector/ # Cloud Run job that pulls live taxi & weather data every 5 mins <br>
├── config/ # Configuration for regional/area mappings, time regressors and ML model <br>
├── etl/ # Flask-based ETL API for transforming and loading into BigQuery <br>
├── streamlit_app/ # Frontend dashboard visualizing trends and forecast
