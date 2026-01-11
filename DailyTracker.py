import streamlit as st
import pandas as pd
import requests
import json
import pickle
import os
import glob
from io import BytesIO
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Daily Tracker Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Minimal CSS ---
st.markdown("""
    <style>
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- API Functions ---
API_URL = "https://dashboard.rabbit-api.app/export"

def fetch_chunk(base, head, payload):
    try:
        r = requests.post(base, headers=head, json=payload, timeout=30)
        if r.status_code == 200:
            try:
                df = pd.read_excel(BytesIO(r.content), engine='openpyxl')
            except:
                df = pd.read_excel(BytesIO(r.content))
            return df
        else:
            st.error(f"API Error {r.status_code}: {r.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

def fetch_day_data(target_date, token):
    date_str = target_date.strftime("%Y-%m-%d")
    start_str = f"{date_str}T00:00:00+02:00"
    end_str = f"{date_str}T23:59:59+02:00"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "module": "RidesSucess",
        "format": "excel",
        "fields": "id,area,start_date_local",
        "filters": json.dumps({
            "startDate": start_str,
            "endDate": end_str
        })
    }

    df = fetch_chunk(API_URL, headers, payload)

    if not df.empty:
        df.columns = df.columns.str.lower().str.strip()
        df.columns = df.columns.str.replace(' ', '_')
        
        col_map = {
            'start_date_local': 'start_date_local',
            'created_at': 'start_date_local',
            'area_name': 'area'
        }
        df = df.rename(columns=col_map)

        if 'start_date_local' in df.columns:
            df['start_date_local'] = pd.to_datetime(df['start_date_local'], errors='coerce')
        
        if 'area' in df.columns:
            df['area'] = df['area'].fillna("Unknown")

    return df

# --- Sidebar & Authentication ---
st.sidebar.title("📊 Daily Tracker")
st.sidebar.markdown("---")

# === AUTHENTICATION LOGIC START ===
# 1. Check Streamlit Secrets first
if "AUTH_TOKEN" in st.secrets:
    AUTH_TOKEN = st.secrets["AUTH_TOKEN"]
    st.sidebar.success("🔓 Authenticated via Secrets")
# 2. Fallback to manual input
else:
    AUTH_TOKEN = st.sidebar.text_input("🔑 API Token", type="password")
# === AUTHENTICATION LOGIC END ===

target_date_input = st.sidebar.date_input("📅 Select Date", datetime.now())

# --- Cache Management Logic ---
CACHE_DIR = ".streamlit_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Define paths based on current date selection
date_str = pd.to_datetime(target_date_input).strftime('%Y-%m-%d')
cache_file = os.path.join(CACHE_DIR, f"data_cache_{date_str}.pkl")
timestamp_file = os.path.join(CACHE_DIR, f"timestamp_{date_str}.txt")

# Check if cache exists right now
cache_exists = os.path.exists(cache_file) and os.path.exists(timestamp_file)

# Sidebar Refresh Button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Force Refresh Data", use_container_width=True):
    st.cache_data.clear()
    
    # Delete physical cache files
    if os.path.exists(CACHE_DIR):
        files = glob.glob(os.path.join(CACHE_DIR, "*"))
        for f in files:
            try:
                os.remove(f)
            except:
                pass
    st.rerun()

# --- Smart Loading Logic ---
# 1. If no cache AND no token, stop here.
if not cache_exists and not AUTH_TOKEN:
    st.warning("👈 Please add your Token to Secrets or enter it in the sidebar.")
    st.stop()

# 2. Variables to hold our data
df_today = pd.DataFrame()
df_yesterday = pd.DataFrame()
df_last_week = pd.DataFrame()
last_refresh_time = datetime.now()

# 3. Try loading from cache first
loaded_from_cache = False
if cache_exists:
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        df_today = cached_data['df_today']
        df_yesterday = cached_data['df_yesterday']
        df_last_week = cached_data['df_last_week']
        
        with open(timestamp_file, 'r') as f:
            t_str = f.read()
            last_refresh_time = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S')
            
        loaded_from_cache = True
    except Exception as e:
        st.error(f"Cache corrupted: {e}")
        # Fall through to fetch logic

# 4. Fetch from API if not loaded from cache
if not loaded_from_cache:
    if not AUTH_TOKEN:
        st.error("Cache missing and no Token provided.")
        st.stop()

    with st.spinner('🔄 Fetching fresh data from API...'):
        today = pd.to_datetime(target_date_input)
        yesterday = today - timedelta(days=1)
        last_week = today - timedelta(days=7)

        df_today = fetch_day_data(today, AUTH_TOKEN)
        df_yesterday = fetch_day_data(yesterday, AUTH_TOKEN)
        df_last_week = fetch_day_data(last_week, AUTH_TOKEN)
        
        # Save to cache
        cached_data = {
            'df_today': df_today,
            'df_yesterday': df_yesterday,
            'df_last_week': df_last_week
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        
        last_refresh_time = datetime.now()
        with open(timestamp_file, 'w') as f:
            f.write(last_refresh_time.strftime('%Y-%m-%d %H:%M:%S'))

# Update Session State for persistent timestamp
st.session_state.last_refresh_time = last_refresh_time


# --- Dashboard Header ---
st.title("📊 Daily Tracker Dashboard")
st.subheader(f"Hourly Ride Analytics • {target_date_input.strftime('%A, %B %d, %Y')}")

col1, col2 = st.columns([4, 1])
with col2:
    if st.button("↻ Refresh View", use_container_width=True):
        st.rerun()
    st.caption(f"📅 Last updated: {st.session_state.last_refresh_time.strftime('%H:%M:%S')}")

st.divider()

# --- Data Processing ---
def create_matrix(df):
    if df.empty or 'start_date_local' not in df.columns:
        return pd.DataFrame()
    
    df = df.dropna(subset=['start_date_local'])
    df['hour'] = df['start_date_local'].dt.hour
    
    pivot = df.pivot_table(
        index='area', 
        columns='hour', 
        values='id', 
        aggfunc='count', 
        fill_value=0
    )
    
    for h in range(24):
        if h not in pivot.columns: 
            pivot[h] = 0
            
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot

# Generate Matrices
matrix_today = create_matrix(df_today)
matrix_yesterday = create_matrix(df_yesterday)
matrix_last_week = create_matrix(df_last_week)

# --- Key Metrics ---
total_today = int(matrix_today.sum().sum()) if not matrix_today.empty else 0
total_yesterday = int(matrix_yesterday.sum().sum()) if not matrix_yesterday.empty else 0
total_last_week = int(matrix_last_week.sum().sum()) if not matrix_last_week.empty else 0

delta_yesterday = total_today - total_yesterday
delta_last_week = total_today - total_last_week
growth_yesterday = (delta_yesterday / total_yesterday * 100) if total_yesterday > 0 else 0
growth_last_week = (delta_last_week / total_last_week * 100) if total_last_week > 0 else 0

# Metrics Display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📈 Total Rides Today",
        f"{total_today:,}",
        f"{delta_yesterday:+,} vs yesterday"
    )

with col2:
    st.metric(
        "📊 Daily Growth",
        f"{growth_yesterday:+.1f}%",
        "vs previous day"
    )

with col3:
    st.metric(
        "📅 Week-over-Week",
        f"{growth_last_week:+.1f}%",
        f"{delta_last_week:+,} rides"
    )

with col4:
    peak_hour = matrix_today.sum(axis=0).idxmax() if not matrix_today.empty else 0
    peak_rides = int(matrix_today.sum(axis=0).max()) if not matrix_today.empty else 0
    st.metric(
        "🔥 Peak Hour",
        f"{peak_hour}:00",
        f"{peak_rides} rides"
    )

st.divider()

# --- Main Content ---
if not matrix_today.empty:
    
    # ========== DETAILED TABLE ==========
    st.subheader("📋 Detailed Hourly Breakdown")
    
    final_view = matrix_today.copy()
    total_hourly = matrix_today.sum(axis=0)
    final_view.loc['Total'] = total_hourly

    yesterday_hourly = matrix_yesterday.sum(axis=0) if not matrix_yesterday.empty else pd.Series(0, index=range(24))
    last_week_hourly = matrix_last_week.sum(axis=0) if not matrix_last_week.empty else pd.Series(0, index=range(24))

    final_view.loc['vs Day Before'] = total_hourly - yesterday_hourly
    final_view.loc['vs Last Week'] = total_hourly - last_week_hourly
    
    # Add totals column
    final_view['Total'] = final_view.sum(axis=1)
    
    # Styling
    def highlight_negatives(val):
        if isinstance(val, (int, float)):
            if val < 0:
                return 'color: red; font-weight: bold'
            elif val > 0:
                return 'color: green; font-weight: bold'
        return ''
    
    hour_cols = [col for col in final_view.columns if col != 'Total']
    gradient_rows = list(matrix_today.index) + ['Total']
    
    try:
        styled = final_view.style\
            .background_gradient(
                cmap="Blues",
                subset=pd.IndexSlice[gradient_rows, hour_cols],
                axis=1
            )\
            .applymap(
                highlight_negatives,
                subset=pd.IndexSlice[['vs Day Before', 'vs Last Week'], hour_cols]
            )\
            .format("{:.0f}")
        
        st.dataframe(styled, use_container_width=True, height=400)
    except Exception as e:
        st.warning(f"Styling unavailable: {e}")
        st.dataframe(final_view, use_container_width=True, height=400)
    
    # Download button
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        csv = final_view.to_csv()
        st.download_button(
            "📥 Download CSV",
            csv,
            file_name=f"rides_{target_date_input.strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.divider()
    
    # ========== HOURLY CHART ==========
    st.subheader("📊 Hourly Ride Distribution")
    
    hourly_data = matrix_today.sum(axis=0)
    yesterday_data = matrix_yesterday.sum(axis=0) if not matrix_yesterday.empty else pd.Series(0, index=range(24))
    last_week_data = matrix_last_week.sum(axis=0) if not matrix_last_week.empty else pd.Series(0, index=range(24))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=hourly_data.values,
        name='Today',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        fill='tozeroy'
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=yesterday_data.values,
        name='Yesterday',
        mode='lines',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=last_week_data.values,
        name='Last Week',
        mode='lines',
        line=dict(color='#2ca02c', width=2, dash='dot')
    ))
    
    fig.update_layout(
        xaxis_title='Hour of Day',
        yaxis_title='Number of Rides',
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"📊 Average: {total_today/24:.1f} rides/hour")
    with col2:
        top_3 = hourly_data.nlargest(3).index.tolist()
        st.info(f"🔥 Busiest: {', '.join([f'{h}:00' for h in top_3])}")

else:
    if cache_exists:
        st.warning("⚠️ Data loaded, but no rides found for the selected date.")
    else:
        st.info("👋 Enter your API Token (or configure Secrets) to start.")

# --- Footer ---
st.divider()
st.caption("Daily Tracker Dashboard v2.2 (Secrets Auth)")
