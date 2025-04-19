import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Reshape, LSTM, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping
import math
import io
import base64
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="PM10 Forecasting - Jakarta Air Quality",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1565C0;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .caption {
        font-size: 0.9rem;
        color: #546E7A;
        font-style: italic;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.5rem solid #1E88E5;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #546E7A;
    }
    .model-explanation {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #757575;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load and preprocess the air quality data"""
    try:
        # Load the data
        # In a real app, you might want to handle file upload or provide a default path
        df = pd.read_excel("2_SORTED_DATA KUALITAS UDARA JAKARTA 2011_2025.xlsx", parse_dates=['tanggal'])
        
        # Preprocess the data
        polls = ['pm10', 'so2', 'co', 'o3', 'no2']
        df[polls] = df[polls].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Filter station and date range
        df = (df[df['stasiun'] == 'DKI1 (Bunderan HI)']
              .loc[df['tanggal'] >= '2014-01-01']
              .sort_values('tanggal')
              .reset_index(drop=True))
        
        # Add cyclical features
        df['doy_sin'] = np.sin(2 * np.pi * df['tanggal'].dt.dayofyear / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['tanggal'].dt.dayofyear / 365)
        df['dow'] = df['tanggal'].dt.dayofweek
        
        # Handle missing values in PM10
        df.loc[df['pm10'] == 0, 'pm10'] = np.nan
        df.set_index('tanggal', inplace=True)
        df['pm10'].fillna(method='ffill', inplace=True)
        df['pm10'].fillna(method='bfill', inplace=True)
        df.reset_index(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Provide sample data if loading fails
        sample_df = pd.DataFrame({
            'tanggal': pd.date_range(start='2014-01-01', periods=100),
            'pm10': np.random.randint(10, 100, 100),
            'stasiun': ['DKI1 (Bunderan HI)'] * 100
        })
        sample_df['doy_sin'] = np.sin(2 * np.pi * sample_df['tanggal'].dt.dayofyear / 365)
        sample_df['doy_cos'] = np.cos(2 * np.pi * sample_df['tanggal'].dt.dayofyear / 365)
        sample_df['dow'] = sample_df['tanggal'].dt.dayofweek
        return sample_df

def create_sequences(df, timesteps, future_steps, n_feats=4):
    """Create sequences for time series prediction"""
    features = df[['pm10', 'doy_sin', 'doy_cos', 'dow']].values
    
    seq, labels, seeds = [], [], []
    for i in range(len(df) - timesteps - future_steps + 1):
        seq.append(features[i:i+timesteps])
        labels.append(df['pm10'].values[i+timesteps : i+timesteps+future_steps])
        seeds.append(df['tanggal'].iloc[i+timesteps])
    
    X_raw = np.array(seq)
    Y_raw = np.array(labels)
    seed_dates = np.array(seeds)
    
    return X_raw, Y_raw, seed_dates

def scale_data(X_raw, Y_raw, n_feats):
    """Scale the data for model training"""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Reshape and scale input features
    X_flat = scaler_X.fit_transform(X_raw.reshape(-1, n_feats))
    Xn = X_flat.reshape((*X_raw.shape, 1))
    
    # Reshape and scale target values
    y_flat = scaler_y.fit_transform(Y_raw.reshape(-1, 1))
    yn = y_flat.reshape((*Y_raw.shape, 1))
    
    return Xn, yn, scaler_X, scaler_y

def build_cnn_lstm_model(timesteps, n_feats, future_steps, filters=32):
    """Build and compile CNN-LSTM model for time series forecasting"""
    # Calculate downsampled timesteps
    down_timesteps = math.ceil(timesteps / 2)
    
    # Encoder
    enc_in = Input(shape=(timesteps, n_feats, 1))
    x = Conv2D(filters, (3, 1), activation='relu', padding='same')(enc_in)
    x = MaxPooling2D((2, 1), padding='same')(x)
    x = Flatten()(x)
    x = Reshape((down_timesteps, n_feats * filters))(x)
    enc_seq, state_h, state_c = LSTM(64, return_sequences=True, return_state=True)(x)
    
    # Decoder
    dec_in = Input(shape=(future_steps, 1))
    d = LSTM(64, return_sequences=True)(dec_in, initial_state=[state_h, state_c])
    d = Dropout(0.3)(d)
    d = LSTM(64, return_sequences=True)(d)
    dec_out = TimeDistributed(Dense(1))(d)
    
    # Build and compile model
    model = Model([enc_in, dec_in], dec_out)
    model.compile(optimizer='adam', loss='mae')
    
    return model

def train_model(model, X_tr, y_tr, epochs=20, batch_size=64, val_split=0.1):
    """Train the model with early stopping"""
    es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    
    with st.spinner('Training model... This may take a while.'):
        history = model.fit(
            [X_tr, y_tr], y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=[es],
            verbose=0
        )
    
    return history

def evaluate_model(model, X_te, y_te, scaler_y):
    """Evaluate model on test data and return metrics"""
    # Make predictions
    pred = model.predict([X_te, y_te], verbose=0)
    
    # Inverse transform predictions and true values
    yt = scaler_y.inverse_transform(y_te[:, :, 0].reshape(-1, 1)).reshape(-1)
    pt = scaler_y.inverse_transform(pred[:, :, 0].reshape(-1, 1)).reshape(-1)
    
    # Calculate metrics
    mape_val = np.mean(np.abs((yt - pt) / yt) * 100)
    rmse_val = np.sqrt(mean_squared_error(yt, pt))
    mae_val = mean_absolute_error(yt, pt)
    r2_val = r2_score(yt, pt)
    
    # Per horizon metrics
    future_steps = y_te.shape[1]
    y_pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).reshape(-1, future_steps)
    y_true = scaler_y.inverse_transform(y_te.reshape(-1, 1)).reshape(-1, future_steps)
    
    horizon_metrics = []
    for h in range(future_steps):
        yt_h = y_true[:, h]
        yp_h = y_pred[:, h]
        mask = (yt_h != 0)
        h_mape = np.mean(np.abs((yt_h[mask] - yp_h[mask]) / yt_h[mask])) * 100
        h_rmse = np.sqrt(mean_squared_error(yt_h[mask], yp_h[mask]))
        horizon_metrics.append((h_mape, h_rmse))
    
    return {
        'mape': mape_val,
        'rmse': rmse_val,
        'mae': mae_val,
        'r2': r2_val,
        'true_values': yt,
        'pred_values': pt,
        'horizon_metrics': horizon_metrics,
        'y_true_full': y_true,
        'y_pred_full': y_pred
    }

def make_forecast(model, df, timesteps, future_steps, scaler_X, scaler_y, start_date=None):
    """Make forecast for future days"""
    if start_date is None:
        # Use the last available data point
        last_X = df[['pm10', 'doy_sin', 'doy_cos', 'dow']].values[-timesteps:]
    else:
        # Filter data up to start_date
        df_filtered = df[df['tanggal'] <= start_date]
        if len(df_filtered) < timesteps:
            st.error(f"Not enough historical data before {start_date} for forecasting.")
            return None, None
        last_X = df_filtered[['pm10', 'doy_sin', 'doy_cos', 'dow']].values[-timesteps:]
    
    # Prepare encoder input
    enc_input = scaler_X.transform(last_X).reshape(1, timesteps, 4, 1)
    
    # Set initial decoder input with the latest actual value
    dec_input = np.zeros((1, future_steps, 1))
    if start_date is None:
        dec_input[0, 0, 0] = scaler_y.transform([[df['pm10'].iloc[-1]]])[0, 0]
    else:
        dec_input[0, 0, 0] = scaler_y.transform([[df_filtered['pm10'].iloc[-1]]])[0, 0]
    
    # Generate autoregressive predictions
    pred = []
    for t in range(future_steps):
        out = model.predict([enc_input, dec_input], verbose=0)
        p = out[0, t, 0]
        pred.append(p)
        
        # Update input for next step
        if t + 1 < future_steps:
            dec_input[0, t + 1, 0] = p
    
    # Inverse transform predictions to original scale
    pred = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
    
    # Generate future dates
    if start_date is None:
        last_date = df['tanggal'].iloc[-1]
    else:
        last_date = pd.to_datetime(start_date)
    
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_steps)
    
    return forecast_dates, pred

def create_anomaly_detection(true_values, pred_values, threshold=1.5):
    """Detect anomalies based on prediction error"""
    residuals = true_values - pred_values
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Define anomalies as points where the residual is beyond threshold * std from mean
    anomalies = np.where(np.abs(residuals - mean_residual) > threshold * std_residual)[0]
    
    return anomalies, residuals

def plot_training_history(history):
    """Plot training and validation loss from model history"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history['loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2)
    ))
    fig.add_trace(go.Scatter(
        y=history.history['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title='Model Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss (MAE)',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_forecast_vs_actual(forecast_dates, forecast_values, actual_dates=None, actual_values=None):
    """Plot forecast against actual values if available"""
    fig = go.Figure()
    
    # Add actual values if available
    if actual_dates is not None and actual_values is not None:
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=actual_values,
            mode='lines',
            name='Actual PM10',
            line=dict(color='#1f77b4', width=2.5)
        ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast PM10',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title='PM10 Concentration Forecast',
        xaxis_title='Date',
        yaxis_title='PM10 Concentration',
        template='plotly_white',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_residuals(residuals, dates=None):
    """Create residual analysis plots"""
    if dates is None:
        dates = np.arange(len(residuals))
    
    # Time series plot of residuals
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=dates,
        y=residuals,
        mode='lines',
        name='Residuals',
        line=dict(color='purple', width=1.5)
    ))
    fig1.add_trace(go.Scatter(
        x=[dates[0], dates[-1]],
        y=[0, 0],
        mode='lines',
        name='Zero Line',
        line=dict(color='black', width=1, dash='dash')
    ))
    
    fig1.update_layout(
        title='Residual Plot (Actual - Predicted)',
        xaxis_title='Time',
        yaxis_title='Residual',
        template='plotly_white',
        height=400
    )
    
    # Histogram of residuals
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker_color='skyblue',
        opacity=0.7,
        name='Frequency'
    ))
    
    # Add KDE (approximated with another histogram trace)
    kde_y, kde_x = np.histogram(residuals, bins=50, density=True)
    kde_x = (kde_x[:-1] + kde_x[1:]) / 2  # Get bin centers
    
    fig2.add_trace(go.Scatter(
        x=kde_x,
        y=kde_y,
        mode='lines',
        name='Density',
        line=dict(color='red', width=2)
    ))
    
    fig2.update_layout(
        title='Distribution of Residuals',
        xaxis_title='Residual Value',
        yaxis_title='Frequency',
        template='plotly_white',
        height=400
    )
    
    return fig1, fig2

def render_cnn_lstm_explanation():
    """Render the CNN-LSTM model explanation section"""
    st.markdown('<div class="model-explanation">', unsafe_allow_html=True)
    
    st.markdown("### How the CNN-LSTM Model Works")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        The CNN-LSTM hybrid model combines Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to capture both spatial and temporal patterns in time series data. Here's how it works:
        
        1. **Input Processing**: The model takes historical PM10 values and temporal features (day of year, day of week) as input.
        
        2. **CNN Encoder**:
           - Applies convolutional filters to extract local patterns in the data
           - Uses pooling to downsample and focus on important features
           - Creates compact representations of the input time series
           
        3. **LSTM Decoder**:
           - Processes the CNN-encoded features sequentially
           - Maintains state information across time steps
           - Captures long-term dependencies in the time series
           - Generates multi-step predictions (7 days ahead)
           
        4. **Teacher Forcing**: During training, the model is provided with the actual values to improve learning.
        
        5. **Autoregressive Forecasting**: For multi-step forecasting, each prediction becomes input for the next step.
        """)
    
    with col2:
        st.markdown("""
        #### Model Architecture
        ```
        CNN Encoder:
        - Conv2D (32 filters)
        - MaxPooling2D
        - Reshape
        - LSTM (64 units)
        
        LSTM Decoder:
        - LSTM (64 units)
        - Dropout (0.3)
        - LSTM (64 units)
        - Dense (1 unit)
        ```
        
        #### Advantages:
        - Captures both temporal patterns and feature interactions
        - Handles multiple input features effectively
        - Generates multi-step forecasts with good accuracy
        - Reduces dimensionality of input data
        """)
    
    st.markdown("""
    #### Use Cases for Air Quality Prediction:
    
    The CNN-LSTM model is particularly well-suited for PM10 prediction because:
    
    - Air quality data has both short-term patterns (daily cycles) and long-term trends (seasonal variations)
    - Multiple factors influence air quality simultaneously (captured by CNN's feature extraction)
    - Historical patterns strongly influence future values (captured by LSTM's memory)
    - The model can adapt to changing patterns over time
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_header():
    """Display the app header and introduction"""
    st.markdown('<h1 class="main-header">🌬️ Jakarta PM10 Air Quality Forecasting</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("""
        <div class="highlight">
        This application provides interactive analysis and forecasting of PM10 air pollutant concentrations
        in Jakarta using a hybrid CNN-LSTM deep learning model. Explore historical trends, analyze prediction accuracy,
        and generate custom forecasts to understand air quality patterns.
        </div>
        """, unsafe_allow_html=True)

def display_data_overview(df):
    """Display data overview section"""
    st.markdown('<h2 class="sub-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">Dataset Information</h3>', unsafe_allow_html=True)
        
        # Display basic dataset information
        st.markdown(f"""
        - **Location**: DKI1 (Bunderan HI), Jakarta
        - **Time Period**: {df['tanggal'].min().strftime('%B %d, %Y')} to {df['tanggal'].max().strftime('%B %d, %Y')}
        - **Total Observations**: {len(df):,}
        - **Pollutants Available**: PM10, SO2, CO, O3, NO2
        """)
        
        # Display data summary statistics
        st.markdown('<h4>PM10 Statistics</h4>', unsafe_allow_html=True)
        stats = df['pm10'].describe().reset_index()
        stats.columns = ['Statistic', 'Value']
        st.dataframe(stats, hide_index=True, use_container_width=True)
    
    with col2:
        # Create histograms of PM10 values
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df['pm10'],
            nbinsx=30,
            marker_color='#1E88E5',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Distribution of PM10 Concentrations',
            xaxis_title='PM10 Value',
            yaxis_title='Frequency',
            template='plotly_white',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive time series plot
    st.markdown('<h3 class="section-header">PM10 Time Series</h3>', unsafe_allow_html=True)
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", df['tanggal'].min().date(), min_value=df['tanggal'].min().date(), max_value=df['tanggal'].max().date())
    with col2:
        end_date = st.date_input("End Date", df['tanggal'].max().date(), min_value=df['tanggal'].min().date(), max_value=df['tanggal'].max().date())
    
    # Filter data based on selected date range
    mask = (df['tanggal'] >= pd.Timestamp(start_date)) & (df['tanggal'] <= pd.Timestamp(end_date))
    filtered_df = df[mask]
    
    # Time series plot with moving average
    fig = go.Figure()
    
    # Add raw PM10 values
    fig.add_trace(go.Scatter(
        x=filtered_df['tanggal'],
        y=filtered_df['pm10'],
        mode='lines',
        name='PM10 Daily',
        line=dict(color='#90CAF9', width=1)
    ))
    
    # Add moving average
    window = st.slider("Moving Average Window (days)", min_value=1, max_value=30, value=7)
    if not filtered_df.empty:
        ma = filtered_df['pm10'].rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=filtered_df['tanggal'],
            y=ma,
            mode='lines',
            name=f'{window}-Day Moving Avg',
            line=dict(color='#1565C0', width=2.5)
        ))
    
    fig.update_layout(
        title='PM10 Concentration Time Series',
        xaxis_title='Date',
        yaxis_title='PM10 Concentration',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data exploration options
    with st.expander("Seasonal Pattern Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week pattern
            dow_avg = df.groupby(df['tanggal'].dt.dayofweek)['pm10'].mean().reset_index()
            dow_avg['tanggal'] = dow_avg['tanggal'].map({
                0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
            })
            
            fig = px.bar(
                dow_avg, 
                x='tanggal', 
                y='pm10',
                title='Average PM10 by Day of Week',
                labels={'tanggal': 'Day of Week', 'pm10': 'Average PM10'},
                color_discrete_sequence=['#1E88E5']
            )
            
            fig.update_layout(height=350, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Monthly pattern
            monthly_avg = df.groupby(df['tanggal'].dt.month)['pm10'].mean().reset_index()
            monthly_avg['Month'] = monthly_avg['tanggal'].map({
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            })
            
            fig = px.bar(
                monthly_avg, 
                x='Month', 
                y='pm10',
                title='Average PM10 by Month',
                labels={'pm10': 'Average PM10'},
                color_discrete_sequence=['#1565C0']
            )
            
            fig.update_layout(height=350, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Sample Data"):
        st.dataframe(filtered_df.head(100), use_container_width=True)

def display_model_section(df):
    """Display model training, evaluation, and forecasting sections"""
    st.markdown('<h2 class="sub-header">🧠 CNN-LSTM Model</h2>', unsafe_allow_html=True)
    
    # Model explanation
    render_cnn_lstm_explanation()
    
    # Model parameter settings
    st.markdown('<h3 class="section-header">Model Configuration</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        timesteps = st.slider("Look-back Window (days)", min_value=7, max_value=60, value=30, 
                              help="Number of past days to consider for prediction")
    
    with col2:
        future_steps = st.slider("Forecast Horizon (days)", min_value=1, max_value=30, value=7,
                                help="Number of days to forecast ahead")
    
    with col3:
        train_split = st.slider("Training Data Fraction", min_value=0.5, max_value=0.9, value=0.8, step=0.05,
                               help="Proportion of data to use for training")
    
    # Create sequences
    n_feats = 4  # pm10, doy_sin, doy_cos, dow
    X_raw, Y_raw, seed_dates = create_sequences(df, timesteps, future_steps, n_feats)
    
    # Scale data
    Xn, yn, scaler_X, scaler_y = scale_data(X_raw, Y_raw, n_feats)
    
    # Train/test split
    split = int(Xn.shape[0] * train_split)
    X_tr, X_te = Xn[:split], Xn[split:]
    y_tr, y_te = yn[:split], yn[split:]
    dates_te = seed_dates[split:]
    
    # Model training section
    st.markdown('<h3 class="section-header">Model Training</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        epochs = st.number_input("Training Epochs", min_value=5, max_value=100, value=20)
    
    with col2:
        batch_size = st.selectbox("Batch Size", options=[16, 32, 64, 128], index=2)
    
    with col3:
        val_split = st.slider("Validation Split", min_value=0.05, max_value=0.3, value=0.1, step=0.05,
                             help="Fraction of training data to use for validation")
    
    # Create/train model button
    train_button = st.button("Train Model", type="primary")
    
    # Session state to store model and results
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = None
    
    # Train model when button is clicked
    if train_button:
        with st.spinner("Building and training CNN-LSTM model..."):
            # Build model
            st.session_state.model = build_cnn_lstm_model(timesteps, n_feats, future_steps)
            
            # Train model
            st.session_state.history = train_model(
                st.session_state.model, 
                X_tr, y_tr, 
                epochs=epochs, 
                batch_size=batch_size, 
                val_split=val_split
            )
            
            # Evaluate model
            st.session_state.evaluation = evaluate_model(
                st.session_state.model, 
                X_te, y_te, 
                scaler_y
            )
            
            st.success("✅ Model training complete!")
    
    # Display training results if model exists
    if st.session_state.model is not None and st.session_state.history is not None:
        # Plot training history
        fig = plot_training_history(st.session_state.history)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<h3 class="section-header">Model Evaluation</h3>', unsafe_allow_html=True)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{st.session_state.evaluation['mape']:.2f}%</div>
                    <div class="metric-label">MAPE</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{st.session_state.evaluation['rmse']:.2f}</div>
                    <div class="metric-label">RMSE</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{st.session_state.evaluation['mae']:.2f}</div>
                    <div class="metric-label">MAE</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{st.session_state.evaluation['r2']:.3f}</div>
                    <div class="metric-label">R²</div>
                </div>""", 
                unsafe_allow_html=True
            )
        
        # Display horizon-specific metrics
        st.markdown('<h4>Forecast Accuracy by Horizon</h4>', unsafe_allow_html=True)
        
        horizon_data = []
        for h, (h_mape, h_rmse) in enumerate(st.session_state.evaluation['horizon_metrics']):
            horizon_data.append({
                'Horizon': f't+{h}',
                'MAPE (%)': h_mape,
                'RMSE': h_rmse
            })
        
        horizon_df = pd.DataFrame(horizon_data)
        st.dataframe(horizon_df, hide_index=True, use_container_width=True)
        
        # Plot actual vs predicted
        st.markdown('<h4>Actual vs Predicted Values</h4>', unsafe_allow_html=True)
        
        # Generate dates for test data visualization
        test_dates = dates_te
        
        # Create interactive plot
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=st.session_state.evaluation['true_values'],
            mode='lines',
            name='Actual PM10',
            line=dict(color='#1f77b4', width=2.5)
        ))
        
        # Add predicted data
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=st.session_state.evaluation['pred_values'],
            mode='lines',
            name='Predicted PM10',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        # Add fill between for visual enhancement
        fig.add_trace(go.Scatter(
            x=np.concatenate([test_dates, test_dates[::-1]]),
            y=np.concatenate([st.session_state.evaluation['true_values'], 
                             st.session_state.evaluation['pred_values'][::-1]]),
            fill='toself',
            fillcolor='rgba(179, 217, 255, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ))
        
        # Improve layout
        fig.update_layout(
            title='Actual vs Predicted PM10 (Test Set)',
            xaxis_title='Date',
            yaxis_title='PM10 Concentration',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual analysis
        with st.expander("Residual Analysis"):
            # Calculate residuals and detect anomalies
            anomalies, residuals = create_anomaly_detection(
                st.session_state.evaluation['true_values'],
                st.session_state.evaluation['pred_values'],
                threshold=1.8
            )
            
            # Plot residuals
            fig1, fig2 = plot_residuals(residuals, test_dates)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Display residual statistics
            st.markdown("#### Residual Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{np.mean(residuals):.2f}")
            
            with col2:
                st.metric("Standard Deviation", f"{np.std(residuals):.2f}")
            
            with col3:
                st.metric("Min", f"{np.min(residuals):.2f}")
            
            with col4:
                st.metric("Max", f"{np.max(residuals):.2f}")
            
            # Display anomalies
            if len(anomalies) > 0:
                st.markdown("#### Detected Anomalies")
                anomaly_data = []
                for idx in anomalies:
                    anomaly_data.append({
                        'Date': test_dates[idx],
                        'Actual': st.session_state.evaluation['true_values'][idx],
                        'Predicted': st.session_state.evaluation['pred_values'][idx],
                        'Error': residuals[idx]
                    })
                
                anomaly_df = pd.DataFrame(anomaly_data)
                st.dataframe(anomaly_df, hide_index=True, use_container_width=True)

def display_forecasting_section(df):
    """Display the forecasting section"""
    st.markdown('<h2 class="sub-header">🔮 PM10 Forecasting</h2>', unsafe_allow_html=True)
    
    # Check if model exists
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first before using the forecasting tools.")
        return
    
    st.markdown("""
    Use the trained CNN-LSTM model to generate PM10 forecasts for future days. The model will use
    historical data up to the selected start date to make predictions.
    """)
    
    # Forecasting options
    col1, col2 = st.columns(2)
    
    with col1:
        # Allow user to select reference date
        max_date = df['tanggal'].max().date()
        forecast_date = st.date_input(
            "Forecast Start Date", 
            value=max_date,
            min_value=df['tanggal'].min().date() + timedelta(days=30),  # Need enough history
            max_value=max_date
        )
    
    with col2:
        forecast_days = st.slider("Forecast Horizon (days)", min_value=7, max_value=60, value=30,
                                 help="Number of days to forecast ahead")
    
    # Generate forecast
    forecast_button = st.button("Generate Forecast", type="primary")
    
    if forecast_button:
        with st.spinner("Generating PM10 forecast..."):
            # Create sequences for the specified time period
            timesteps = 30  # Use 30-day look-back as in original code
            n_feats = 4     # pm10, doy_sin, doy_cos, dow
            
            # Make forecast
            forecast_dates, forecast_values = make_forecast(
                st.session_state.model,
                df,
                timesteps,
                forecast_days,
                st.session_state.scaler_X,
                st.session_state.scaler_y,
                start_date=pd.Timestamp(forecast_date)
            )
            
            if forecast_dates is not None:
                # Store forecast in session state
                st.session_state.forecast_dates = forecast_dates
                st.session_state.forecast_values = forecast_values
                
                # Get actual data for comparison plot
                history_start = pd.Timestamp(forecast_date) - timedelta(days=60)
                history_mask = (df['tanggal'] >= history_start) & (df['tanggal'] <= pd.Timestamp(forecast_date))
                history_df = df[history_mask]
                
                # Plot forecast
                fig = plot_forecast_vs_actual(
                    forecast_dates,
                    forecast_values,
                    history_df['tanggal'],
                    history_df['pm10']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add AQI interpretation
                with st.expander("Air Quality Index (AQI) Interpretation"):
                    st.markdown("""
                    ### PM10 Air Quality Index (AQI) Categories
                    
                    | PM10 Range (µg/m³) | AQI Category | Interpretation |
                    |-------------------|-------------|----------------|
                    | 0-50 | Good | Air quality is considered satisfactory, and air pollution poses little or no risk |
                    | 51-100 | Moderate | Air quality is acceptable; however, some pollutants may be a concern for a very small number of people |
                    | 101-250 | Unhealthy for Sensitive Groups | Members of sensitive groups may experience health effects |
                    | 251-350 | Unhealthy | Everyone may begin to experience health effects |
                    | 351-420 | Very Unhealthy | Health warnings of emergency conditions; entire population is more likely to be affected |
                    | >420 | Hazardous | Health alert: everyone may experience more serious health effects |
                    """)
                    
                    # Calculate AQI categories for forecast
                    def get_aqi_category(pm10):
                        if pm10 <= 50:
                            return "Good"
                        elif pm10 <= 100:
                            return "Moderate"
                        elif pm10 <= 250:
                            return "Unhealthy for Sensitive Groups"
                        elif pm10 <= 350:
                            return "Unhealthy"
                        elif pm10 <= 420:
                            return "Very Unhealthy"
                        else:
                            return "Hazardous"
                    
                    aqi_categories = [get_aqi_category(val) for val in forecast_values]
                    aqi_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'PM10 Forecast': [round(val, 1) for val in forecast_values],
                        'AQI Category': aqi_categories
                    })
                    
                    st.dataframe(aqi_df, hide_index=True, use_container_width=True)
                
                # Download forecasts
                csv = aqi_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast Data",
                    data=csv,
                    file_name=f"pm10_forecast_{forecast_date}.csv",
                    mime="text/csv"
                )

def display_insights_section():
    """Display insights and recommendations based on forecasts"""
    st.markdown('<h2 class="sub-header">📈 Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'forecast_values') or st.session_state.forecast_values is None:
        st.info("Generate a forecast to see insights and recommendations.")
        return
    
    # Calculate some metrics from the forecast
    mean_forecast = np.mean(st.session_state.forecast_values)
    max_forecast = np.max(st.session_state.forecast_values)
    max_date = st.session_state.forecast_dates[np.argmax(st.session_state.forecast_values)]
    
    # Define AQI thresholds
    # Determine the predominant AQI category
    def get_aqi_category(pm10):
        if pm10 <= 50:
            return "Good"
        elif pm10 <= 100:
            return "Moderate"
        elif pm10 <= 250:
            return "Unhealthy for Sensitive Groups"
        elif pm10 <= 350:
            return "Unhealthy"
        elif pm10 <= 420:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    aqi_categories = [get_aqi_category(val) for val in st.session_state.forecast_values]
    predominant_aqi = max(set(aqi_categories), key=aqi_categories.count)
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Forecast Summary</h3>', unsafe_allow_html=True)
        
        st.markdown(f"""
        #### Key Metrics:
        - **Average PM10 Forecast**: {mean_forecast:.1f} µg/m³
        - **Maximum PM10 Forecast**: {max_forecast:.1f} µg/m³ on {max_date.strftime('%B %d, %Y')}
        - **Predominant Air Quality**: {predominant_aqi}
        
        #### Trend Analysis:
        - {"🔺 Increasing trend" if np.polyfit(range(len(st.session_state.forecast_values)), st.session_state.forecast_values, 1)[0] > 0 else "🔻 Decreasing trend"} in PM10 concentrations over the forecast period
        - {"⚠️ Potential air quality event detected" if max_forecast > 150 else "✅ No significant air quality events expected"}
        """)
    
    with col2:
        st.markdown('<h3 class="section-header">Recommendations</h3>', unsafe_allow_html=True)
        
        if predominant_aqi in ["Good", "Moderate"]:
            st.markdown("""
            #### General Population:
            - Continue normal outdoor activities
            - No special precautions needed
            
            #### Sensitive Groups:
            - Monitor air quality if you have respiratory conditions
            - Carry medication if you have asthma
            """)
        elif predominant_aqi == "Unhealthy for Sensitive Groups":
            st.markdown("""
            #### General Population:
            - Consider reducing prolonged outdoor exertion
            - Monitor air quality reports
            
            #### Sensitive Groups:
            - Reduce prolonged or heavy outdoor exertion
            - Take more breaks during outdoor activities
            - Reschedule outdoor activities to when air quality improves
            """)
        else:
            st.markdown("""
            #### General Population:
            - Reduce prolonged or heavy outdoor exertion
            - Consider moving activities indoors
            - Use air purifiers when indoors
            
            #### Sensitive Groups:
            - Avoid all outdoor physical activities
            - Remain indoors and keep activity levels low
            - Run air purifiers and close windows
            - Wear N95 masks if going outdoors is necessary
            """)
    
    # Air quality calendar view
    st.markdown('<h3 class="section-header">Air Quality Calendar View</h3>', unsafe_allow_html=True)
    
    # Create a calendar-like visualization of the forecast
    dates = st.session_state.forecast_dates
    values = st.session_state.forecast_values
    
    # Group by week
    week_starts = [d.date() for d in dates if d.weekday() == 0]
    if dates[0].weekday() != 0:
        week_starts = [dates[0].date()] + week_starts
    
    calendar_data = []
    for d, v in zip(dates, values):
        calendar_data.append({
            'date': d.date(),
            'day': d.day,
            'weekday': d.strftime('%a'),
            'value': v,
            'category': get_aqi_category(v),
            'week_number': (d.date() - week_starts[0]).days // 7
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    # Define colors for AQI categories
    color_map = {
        'Good': '#8BC34A',
        'Moderate': '#FFEB3B',
        'Unhealthy for Sensitive Groups': '#FF9800',
        'Unhealthy': '#F44336',
        'Very Unhealthy': '#9C27B0',
        'Hazardous': '#7B1FA2'
    }
    
    # Create a grid of weeks
    weeks = calendar_df['week_number'].unique()
    
    for week in weeks:
        week_data = calendar_df[calendar_df['week_number'] == week]
        
        cols = st.columns(7)
        for i, col in enumerate(cols):
            try:
                day_data = week_data[week_data['date'].dt.weekday == i].iloc[0]
                
                col.markdown(
                    f"""<div style="background-color: {color_map[day_data['category']]}; 
                           color: {'black' if day_data['category'] in ['Good', 'Moderate'] else 'white'}; 
                           padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-size: 0.8rem;">{day_data['weekday']}</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{day_data['day']}</div>
                        <div style="font-size: 0.9rem;">{day_data['value']:.1f}</div>
                    </div>""",
                    unsafe_allow_html=True
                )
            except (IndexError, KeyError):
                # Empty cell for days not in the forecast
                col.markdown(
                    """<div style="background-color: #E0E0E0; 
                           padding: 10px; border-radius: 5px; text-align: center;">
                        <div style="font-size: 0.8rem;">&nbsp;</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">&nbsp;</div>
                        <div style="font-size: 0.9rem;">&nbsp;</div>
                    </div>""",
                    unsafe_allow_html=True
                )
    
    # Display legend
    st.markdown("#### Air Quality Categories Legend:")
    
    legend_cols = st.columns(len(color_map))
    for i, (category, color) in enumerate(color_map.items()):
        legend_cols[i].markdown(
            f"""<div style="background-color: {color}; 
                   color: {'black' if category in ['Good', 'Moderate'] else 'white'}; 
                   padding: 5px; border-radius: 3px; text-align: center; font-size: 0.8rem;">
                {category}
            </div>""",
            unsafe_allow_html=True
        )

def main():
    """Main function to run the Streamlit app"""
    # Initialize session state variables
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = None
    if 'forecast_dates' not in st.session_state:
        st.session_state.forecast_dates = None
    if 'forecast_values' not in st.session_state:
        st.session_state.forecast_values = None
    if 'scaler_X' not in st.session_state:
        st.session_state.scaler_X = None
    if 'scaler_y' not in st.session_state:
        st.session_state.scaler_y = None
    
    # Display the header
    display_header()
    
    # Create tabs for the main sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Data Exploration", 
        "🧠 Model Training", 
        "🔮 Forecasting", 
        "📈 Insights"
    ])
    
    # Load and preprocess data
    with st.spinner('Loading data...'):
        df = load_data()
    
    # Fill each tab with content
    with tab1:
        display_data_overview(df)
    
    with tab2:
        display_model_section(df)
    
    with tab3:
        display_forecasting_section(df)
    
    with tab4:
        display_insights_section()
    
    # Footer
    st.markdown(
        """<div class="footer">
            Developed by AI & ML Experts | Jakarta Air Quality Analysis | 2025
        </div>""",
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    main()