import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import io

def display_enhanced_header():
    """Display an enhanced app header with Jakarta skyline and pollution visualization"""
    
    # CSS for the enhanced homepage
    st.markdown("""
    <style>
        /* Main container styling */
        .homepage-container {
            padding: 0;
            margin: 0;
            width: 100%;
        }
        
        /* Hero section with background image */
        .hero-section {
            position: relative;
            height: 500px;
            background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                              url('https://storage.googleapis.com/aipi-models/placeholder/jakarta_skyline.jpg');
            background-size: cover;
            background-position: center;
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 2rem;
            padding: 2rem;
        }
        
        /* Fallback if image doesn't load */
        @media only screen and (max-width: 1px) {
            .hero-section {
                background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            }
        }
        
        /* Main title styling */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
        }
        
        /* Subtitle styling */
        .hero-subtitle {
            font-size: 1.5rem;
            font-weight: 400;
            margin-bottom: 2rem;
            max-width: 800px;
            line-height: 1.5;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
        }
        
        /* Button styling */
        .hero-button {
            display: inline-block;
            background-color: #1E88E5;
            color: white;
            padding: 0.8rem 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.3s ease;
            margin: 0.5rem;
            border: none;
        }
        
        .hero-button:hover {
            background-color: #0D47A1;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        
        /* Info cards container */
        .info-cards {
            display: flex;
            justify-content: space-between;
            margin: 2rem 0;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        /* Individual info card */
        .info-card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            padding: 1.5rem;
            flex: 1;
            min-width: 250px;
            transition: transform 0.3s ease;
        }
        
        .info-card:hover {
            transform: translateY(-5px);
        }
        
        /* Card icon */
        .card-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
            color: #1E88E5;
        }
        
        /* Card title */
        .card-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #0D47A1;
        }
        
        /* Card content */
        .card-content {
            color: #546E7A;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        /* Feature section */
        .feature-section {
            margin: 3rem 0;
            background-color: #f6f9fc;
            padding: 2rem;
            border-radius: 10px;
        }
        
        /* Feature section title */
        .section-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 2rem;
            color: #0D47A1;
            text-align: center;
        }
        
        /* Feature card */
        .feature-card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        }
        
        /* AQI index container */
        .aqi-container {
            margin: 3rem 0;
            text-align: center;
        }
        
        /* AQI index scale */
        .aqi-scale {
            display: flex;
            margin-top: 1rem;
            border-radius: 50px;
            overflow: hidden;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        /* AQI category */
        .aqi-category {
            flex: 1;
            padding: 0.8rem;
            color: white;
            font-weight: 600;
        }
        
        /* Footer styling */
        .homepage-footer {
            margin-top: 4rem;
            padding: 2rem;
            background-color: #f6f9fc;
            border-radius: 10px;
            text-align: center;
            color: #546E7A;
        }
        
        /* Animation for fading in elements */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-in {
            animation: fadeIn 1s ease forwards;
        }
        
        /* Styling for the Jakarta map visualization */
        .map-container {
            margin: 2rem 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }
        
        /* Styling for the PM10 trend visualization */
        .trend-container {
            margin: 2rem 0;
            padding: 1rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        
        /* Styling for tab headers */
        div[data-baseweb="tab-list"] {
            background-color: white !important;
            border-radius: 10px !important;
            padding: 0.5rem !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
        }
        
        div[data-baseweb="tab"] {
            font-weight: 600 !important;
            color: #546E7A !important;
        }
        
        div[data-baseweb="tab"][aria-selected="true"] {
            color: #1E88E5 !important;
            background-color: #E3F2FD !important;
            border-radius: 8px !important;
        }
        
        /* Styling for the latest reading container */
        .latest-reading {
            background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
            padding: 2rem;
            border-radius: 10px;
            margin: 2rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
        
        .reading-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #0D47A1;
            margin-bottom: 1rem;
        }
        
        .reading-value {
            font-size: 3rem;
            font-weight: 800;
            color: #1565C0;
        }
        
        .reading-unit {
            font-size: 1.2rem;
            color: #1976D2;
        }
        
        .reading-time {
            font-size: 0.9rem;
            color: #546E7A;
            margin-top: 0.5rem;
        }
        
        /* Styling for the pollution info box */
        .pollution-info {
            background-color: white;
            border-left: 5px solid #1E88E5;
            padding: 1.5rem;
            margin: 2rem 0;
            border-radius: 0.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero section with Jakarta skyline
    st.markdown("""
    <div class="homepage-container">
        <div class="hero-section">
            <div class="hero-title">Jakarta Air Quality Monitor</div>
            <div class="hero-subtitle">A comprehensive dashboard for monitoring, analyzing, and forecasting PM10 air pollution levels in Jakarta using advanced machine learning techniques</div>
            <div>
                <button class="hero-button" onclick="document.querySelectorAll('.stTabs button')[0].click();">Explore Data</button>
                <button class="hero-button" onclick="document.querySelectorAll('.stTabs button')[2].click();" style="background-color: #4CAF50;">View Forecast</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_latest_readings(df):
    """Display latest PM10 readings with visual indicators"""
    # Get the latest reading
    latest_date = df['tanggal'].max()
    latest_reading = df[df['tanggal'] == latest_date]['pm10'].values[0]
    
    # Determine AQI category and color
    if latest_reading <= 50:
        category = "Good"
        color = "#8BC34A"
    elif latest_reading <= 100:
        category = "Moderate"
        color = "#FFEB3B"
    elif latest_reading <= 250:
        category = "Unhealthy for Sensitive Groups"
        color = "#FF9800"
    elif latest_reading <= 350:
        category = "Unhealthy"
        color = "#F44336"
    elif latest_reading <= 420:
        category = "Very Unhealthy"
        color = "#9C27B0"
    else:
        category = "Hazardous"
        color = "#7B1FA2"
    
    st.markdown(f"""
    <div class="latest-reading" style="border-left: 10px solid {color};">
        <div class="reading-title">Latest PM10 Reading at Bundaran HI</div>
        <div class="reading-value">{latest_reading:.1f} <span class="reading-unit">µg/m³</span></div>
        <div style="font-size: 1.3rem; font-weight: 600; color: {color};">{category}</div>
        <div class="reading-time">Last updated: {latest_date.strftime('%B %d, %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

def display_info_cards():
    """Display information cards about air quality and the app"""
    st.markdown("""
    <div class="info-cards">
        <div class="info-card">
            <div class="card-icon">🏙️</div>
            <div class="card-title">Jakarta's Air Challenge</div>
            <div class="card-content">
                Jakarta, with a population of over 10 million, faces significant air quality challenges due to vehicular emissions, industrial activities, and construction. PM10 particles are among the most critical pollutants affecting the city's air quality.
            </div>
        </div>
        
        <div class="info-card">
            <div class="card-icon">🧠</div>
            <div class="card-title">AI-Powered Forecasting</div>
            <div class="card-content">
                Our dashboard uses a state-of-the-art CNN-LSTM hybrid model to analyze historical patterns and predict future PM10 levels with high accuracy, helping residents and authorities plan ahead.
            </div>
        </div>
        
        <div class="info-card">
            <div class="card-icon">📊</div>
            <div class="card-title">Interactive Analytics</div>
            <div class="card-content">
                Explore historical trends, seasonal patterns, and real-time forecasts through interactive visualizations designed to provide clear insights into Jakarta's air quality situation.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_aqi_scale():
    """Display AQI scale with color indicators"""
    st.markdown("""
    <div class="aqi-container">
        <div class="section-title">Air Quality Index (AQI) Scale</div>
        <div class="aqi-scale">
            <div class="aqi-category" style="background-color: #8BC34A;">Good<br>(0-50)</div>
            <div class="aqi-category" style="background-color: #FFEB3B; color: black;">Moderate<br>(51-100)</div>
            <div class="aqi-category" style="background-color: #FF9800;">Unhealthy for Sensitive Groups<br>(101-250)</div>
            <div class="aqi-category" style="background-color: #F44336;">Unhealthy<br>(251-350)</div>
            <div class="aqi-category" style="background-color: #9C27B0;">Very Unhealthy<br>(351-420)</div>
            <div class="aqi-category" style="background-color: #7B1FA2;">Hazardous<br>(>420)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_jakarta_map():
    """Display Jakarta map with monitoring stations"""
    st.markdown("""
    <div class="map-container">
        <div class="section-title">Jakarta Air Quality Monitoring Stations</div>
        <img src="https://storage.googleapis.com/aipi-models/placeholder/jakarta_air_quality_map.jpg" 
             alt="Jakarta Air Quality Monitoring Stations Map" 
             style="width: 100%; border-radius: 10px;" />
    </div>
    """, unsafe_allow_html=True)

def display_pm10_info():
    """Display information about PM10 and its health effects"""
    st.markdown("""
    <div class="pollution-info">
        <h3>What is PM10?</h3>
        <p>PM10 refers to particulate matter (PM) with a diameter of 10 micrometers or less. These particles are small enough to be inhaled and can cause significant health issues, especially for sensitive populations like children, elderly, and those with respiratory conditions.</p>
        
        <h4>Health Effects:</h4>
        <ul>
            <li>Respiratory symptoms such as irritation of airways, coughing, difficulty breathing</li>
            <li>Decreased lung function and increased respiratory infections</li>
            <li>Aggravated asthma and chronic bronchitis</li>
            <li>Premature death in people with heart or lung disease</li>
        </ul>
        
        <h4>Major Sources in Jakarta:</h4>
        <ul>
            <li>Vehicle emissions (particularly from older diesel vehicles)</li>
            <li>Industrial activities</li>
            <li>Construction dust</li>
            <li>Road dust resuspension</li>
            <li>Open burning of waste</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_pm10_trend(df):
    """Display PM10 trend visualization"""
    # Prepare data for trend visualization
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly['tanggal'].dt.to_period('M')
    monthly_avg = df_monthly.groupby('month')['pm10'].mean().reset_index()
    monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()
    
    # Create trend visualization
    fig = px.line(
        monthly_avg, 
        x='month', 
        y='pm10',
        title='Monthly Average PM10 Concentrations',
        labels={'month': 'Month', 'pm10': 'PM10 Concentration (µg/m³)'},
        line_shape='spline',
        template='plotly_white'
    )
    
    fig.update_traces(line=dict(color='#1E88E5', width=4))
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            title_font=dict(size=14),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            title_font=dict(size=14),
        ),
        title=dict(
            font=dict(size=18, color='#0D47A1')
        )
    )
    
    # Add a reference line for moderate air quality threshold
    fig.add_hline(
        y=100, 
        line_dash="dash", 
        line_color="#FF9800",
        annotation_text="Moderate AQI Threshold",
        annotation_position="top right"
    )
    
    st.markdown('<div class="trend-container">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_features_section():
    """Display key features of the dashboard"""
    st.markdown("""
    <div class="feature-section">
        <div class="section-title">Key Features</div>
        
        <div class="feature-card">
            <div style="margin-right: 1.5rem; color: #1E88E5; font-size: 2rem;">📊</div>
            <div>
                <div style="font-weight: 700; font-size: 1.2rem; color: #0D47A1; margin-bottom: 0.5rem;">Interactive Data Exploration</div>
                <div style="color: #546E7A;">Visualize historical PM10 data with interactive charts and filters. Analyze trends, seasonal patterns, and explore relationships between different time periods.</div>
            </div>
        </div>
        
        <div class="feature-card">
            <div style="margin-right: 1.5rem; color: #1E88E5; font-size: 2rem;">🧠</div>
            <div>
                <div style="font-weight: 700; font-size: 1.2rem; color: #0D47A1; margin-bottom: 0.5rem;">Advanced ML Forecasting</div>
                <div style="color: #546E7A;">Utilize a hybrid CNN-LSTM deep learning model that captures both spatial and temporal patterns in the data to generate accurate multi-step forecasts of PM10 levels.</div>
            </div>
        </div>
        
        <div class="feature-card">
            <div style="margin-right: 1.5rem; color: #1E88E5; font-size: 2rem;">📈</div>
            <div>
                <div style="font-weight: 700; font-size: 1.2rem; color: #0D47A1; margin-bottom: 0.5rem;">Actionable Insights</div>
                <div style="color: #546E7A;">Receive health recommendations based on forecasted air quality levels. View air quality calendar and understand potential risks for sensitive groups.</div>
            </div>
        </div>
        
        <div class="feature-card">
            <div style="margin-right: 1.5rem; color: #1E88E5; font-size: 2rem;">🔍</div>
            <div>
                <div style="font-weight: 700; font-size: 1.2rem; color: #0D47A1; margin-bottom: 0.5rem;">Anomaly Detection</div>
                <div style="color: #546E7A;">Automatically identify unusual pollution events and outliers in the data, helping to spot emerging air quality issues before they become severe.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_footer():
    """Display footer section"""
    st.markdown("""
    <div class="homepage-footer">
        <div style="font-size: 1.2rem; font-weight: 600; color: #0D47A1; margin-bottom: 1rem;">Jakarta Air Quality Monitoring</div>
        <div style="margin-bottom: 1rem;">An advanced air quality monitoring and forecasting platform for Jakarta</div>
        <div style="font-size: 0.9rem;">© 2025 AI & ML Experts | Data Sources: Jakarta Environmental Agency</div>
    </div>
    """, unsafe_allow_html=True)

def display_homepage(df):
    """Main function to display the enhanced homepage"""
    # Display the enhanced header with Jakarta skyline
    display_enhanced_header()
    
    # Display latest PM10 readings
    display_latest_readings(df)
    
    # Display info cards
    display_info_cards()
    
    # Display PM10 trend visualization
    display_pm10_trend(df)
    
    # Create two columns for the map and AQI scale
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Display Jakarta map with monitoring stations
        display_jakarta_map()
    
    with col2:
        # Display AQI scale
        display_aqi_scale()
        
        # Display PM10 information
        display_pm10_info()
    
    # Display key features section
    display_features_section()
    
    # Display footer
    display_footer()

# This function would be called from the main.py file
def show_homepage():
    # Load data
    try:
        df = pd.read_excel("2_SORTED_DATA KUALITAS UDARA JAKARTA 2011_2025.xlsx", parse_dates=['tanggal'])
        # Basic preprocessing for homepage
        df = df[df['stasiun'] == 'DKI1 (Bunderan HI)']
        df = df.sort_values('tanggal')
    except Exception as e:
        # Create sample data if loading fails
        start_date = pd.Timestamp('2014-01-01')
        end_date = pd.Timestamp('2023-12-31')
        date_range = pd.date_range(start=start_date, end=end_date)
        
        np.random.seed(42)  # For reproducibility
        
        # Create seasonal pattern with some randomness
        seasonal_pattern = np.sin(np.linspace(0, 8*np.pi, len(date_range))) * 30 + 70
        random_variation = np.random.normal(0, 15, len(date_range))
        pm10_values = seasonal_pattern + random_variation
        
        df = pd.DataFrame({
            'tanggal': date_range,
            'pm10': pm10_values,
            'stasiun': ['DKI1 (Bunderan HI)'] * len(date_range)
        })
    
    # Display the homepage with the loaded/sample data
    display_homepage(df)

# For testing in isolation
if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="PM10 Forecasting - Jakarta Air Quality",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Show the homepage
    show_homepage()