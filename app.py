# app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Walmart Sales Dashboard")

st.title("🛍️ Walmart Sales Prediction Dashboard")
st.write("This app visualizes the performance of an XGBoost model on test data.")

# --- Load Data (with caching) ---
@st.cache
def load_data():
    data = pd.read_csv('C:/Users/USER/Skill Dev/Main Porto/test_results_for_dashboard.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Select Filters")

# 1. Store Selector
store = st.sidebar.selectbox(
    'Select Store',
    df['Store'].unique(),
    index=0 # Default to the first store
)

# 2. Dept Selector (depends on store)
valid_depts = np.sort(df[df['Store'] == store]['Dept'].unique())
dept = st.sidebar.selectbox(
    'Select Department',
    valid_depts,
    index=0 # Default to the first dept
)

# --- Main Panel ---
st.header(f"Sales Forecast for Store {store}, Department {dept}")

# Filter data based on sidebar selection
plot_data = df[(df['Store'] == store) & (df['Dept'] == dept)].sort_values('Date')

if plot_data.empty:
    st.error("No data available for this selection.")
else:
    # --- Summary Metrics ---
    # Calculate metrics for this specific view
    rmse = np.sqrt(mean_squared_error(plot_data['Actual_Sales'], plot_data['Predicted_Sales']))
    
    # Use our robust MAPE calculation
    epsilon = 1e-6
    mape = np.mean(np.abs((plot_data['Actual_Sales'] - plot_data['Predicted_Sales']) / 
                         (plot_data['Actual_Sales'] + epsilon))) * 100
    
    st.write("Model Performance (This View):")
    col1, col2 = st.columns(2)
    col1.metric("Model RMSE", f"${rmse:,.2f}")
    col2.metric("Model MAPE", f"{mape:.2f}%")

    # --- Plotly Chart ---
    fig = go.Figure()
    
    # Actual Sales
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['Actual_Sales'],
        mode='lines+markers',
        name='Actual Sales',
        line=dict(color='blue')
    ))
    
    # Predicted Sales
    fig.add_trace(go.Scatter(
        x=plot_data['Date'],
        y=plot_data['Predicted_Sales'],
        mode='lines+markers',
        name='Predicted Sales (XGBoost)',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Actual Sales vs. Predicted Sales',
        xaxis_title='Date',
        yaxis_title='Weekly Sales ($)',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- Raw Data Table ---
    st.subheader("Raw Data (This View)")
    st.dataframe(plot_data[['Date', 'Store', 'Dept', 'Actual_Sales', 'Predicted_Sales', 'Temperature', 'IsHoliday']].tail(10))