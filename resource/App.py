import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

st.set_page_config(page_title="Prediksi Kurs Jual Rupiah", layout="wide", page_icon="💰")

st.markdown(
    """
    <style>
        .stApp {background-color: #F8F9FA;}
        .sidebar .sidebar-content {background-color: #FF6F61; color: white;}
        .stButton>button {background-color: #28A745; color: white; font-weight: bold; border-radius: 8px;}
        .stButton>button:hover {background-color: #218838;}
    </style>
    """,
    unsafe_allow_html=True
)

# 📌 Load Data dari File Excel
def load_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_excel(uploaded_file)
        df = df.iloc[3:].reset_index(drop=True)
        df.columns = ["NO", "Nilai", "Kurs Jual", "Kurs Beli", "Tanggal"]
        df = df[["Tanggal", "Kurs Jual"]]
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors='coerce')
        df = df.dropna(subset=["Tanggal"])
        df["Kurs Jual"] = df["Kurs Jual"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df["Kurs Jual"] = pd.to_numeric(df["Kurs Jual"], errors='coerce')
        df = df.dropna(subset=["Kurs Jual"])
        df.set_index("Tanggal", inplace=True)
        df.sort_index(ascending=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        return None

# 🔥 Train Model ARIMA
def train_model(df):
    if len(df) < 10:
        st.error("Data terlalu sedikit untuk membangun model ARIMA. Pastikan minimal ada 10 data.")
        return None
    with st.spinner("🔄 Melatih model ARIMA..."):
        time.sleep(2)
        model = ARIMA(df["Kurs Jual"], order=(1,1,1))
        model_fit = model.fit()
    return model_fit

# 📊 Prediksi Kurs Jual Menggunakan ARIMA
def predict(start, end, df, model_fit):
    selisih_hari = (end - start).days + 1
    forecast = model_fit.forecast(steps=selisih_hari).to_numpy().ravel()

    last_date = df.index[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, selisih_hari + 1)]
    forecast_df = pd.DataFrame({"Tanggal": forecast_dates, "Prediksi Kurs Jual": forecast})
    forecast_df.set_index("Tanggal", inplace=True)
    
    st.subheader(f"📈 Data Prediksi {selisih_hari} Hari Kedepan")
    with st.expander("🔍 Lihat Detail Prediksi"):
        st.dataframe(forecast_df, width=600)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Kurs Jual"], mode='lines+markers', name='Data Historis', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Prediksi Kurs Jual"], mode='lines+markers', name='Prediksi', line=dict(color='red', dash='dash')))
    fig.update_layout(title="Prediksi Kurs Jual dengan ARIMA", xaxis_title="Tanggal", yaxis_title="Kurs Jual", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# 📉 Evaluasi Model
def evaluate_model(df, model_fit):
    forecast_steps = min(len(df), 30)
    actual = df["Kurs Jual"].iloc[-forecast_steps:]
    predicted = model_fit.forecast(steps=forecast_steps).to_numpy().ravel()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    st.metric(label="📊 Mean Absolute Error (MAE)", value=f"{mae:.2f}")
    st.metric(label="📉 Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")

# 💹 Tampilan Utama
st.title("💹 Prediksi Kurs Jual Rupiah Terhadap Mata Uang Amerika Serikat")
st.markdown("---")
st.sidebar.header("⚙️ Pengaturan")

# 📂 Upload File
uploaded_files = st.sidebar.file_uploader("📂 Pilih File Excel", type=".xlsx")

if uploaded_files is not None:
    df = load_data(uploaded_files)

    if df is None or df.empty:
        st.error("File tidak valid atau tidak mengandung data yang diperlukan.")
    else:
        with st.expander("📌 Data yang Diupload", expanded=True):
            st.dataframe(df, width=800, height=400)
        
        model_fit = train_model(df)
        if model_fit:
            evaluate_model(df, model_fit)
            start_date = df.index[-1] if not df.empty else None
            start = st.sidebar.date_input("📅 Tanggal Mulai", value=start_date, disabled=True)
            end = st.sidebar.date_input("📅 Tanggal Selesai", value=None)
            start = pd.to_datetime(start)
            end = pd.to_datetime(end) if end else None
            
            button = st.sidebar.button("🔮 Prediksi!", type="primary")

            if button:
                if end is not None:
                    if end > start:
                        predict(start, end, df, model_fit)
                    else:
                        st.warning("Tanggal selesai harus lebih besar dari tanggal mulai!")
                else:
                    st.warning("Isi terlebih dahulu tanggal selesai!")
