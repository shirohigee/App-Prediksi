import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Prediksi Kurs Jual Rupiah", layout="wide", page_icon="💰")

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

# 🔥 Fungsi Train Model dengan Train-Test Split dan Walk-Forward Validation
def train_and_evaluate_model(df):
    if len(df) < 20:
        st.error("Data terlalu sedikit untuk membangun model ARIMA. Pastikan minimal ada 20 data.")
        return None, None, None

    # Pisahkan Data (80% Train - 20% Test)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Latih Model ARIMA
    model = ARIMA(train['Kurs Jual'], order=(2, 1, 2))
    model_fit = model.fit()

    # Walk-Forward Validation
    history = list(train['Kurs Jual'])
    predictions = []
    actual_values = list(test['Kurs Jual'])

    for t in range(len(test)):
        model = ARIMA(history, order=(2, 1, 2))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(actual_values[t])

    # Hitung Error
    mae = float(mean_absolute_error(actual_values, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual_values, predictions)))
    mean_actual = np.mean(actual_values)
    mae_percentage = (mae / mean_actual) * 100
    rmse_percentage = (rmse / mean_actual) * 100


    return model_fit, mae, rmse, mae_percentage, rmse_percentage

# 📊 Prediksi Kurs Jual Menggunakan ARIMA
def predict(start, end, df, model_fit):
    selisih_hari = (end - start).days + 1
    forecast = model_fit.forecast(steps=selisih_hari)
    forecast = np.array(forecast).ravel()

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
def evaluate_model(mae, rmse, mae_percentage, rmse_percentage):
    # st.metric(label="📊 Mean Absolute Error (MAE)", value=f"{mae:.2f}")
    # st.metric(label="📉 Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
    st.metric(label="📊 MAE Mean Absolute Error", value=f"{mae_percentage:.4f}")
    st.metric(label="📉 Root Mean Squared Error (RMSE)", value=f"{rmse_percentage:.4f}")

# 💹 Tampilan Utama
st.title("💹 Prediksi Kurs Jual Rupiah Terhadap Dollar Amerika Serikat")
st.markdown("---")
st.sidebar.header("⚙️ Pengaturan")

# 📂 Upload File
uploaded_file = st.sidebar.file_uploader("📂 Pilih File Excel", type=".xlsx")

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is None or df.empty:
        st.error("File tidak valid atau tidak mengandung data yang diperlukan.")
    else:
        st.session_state["df"] = df  # Simpan Data di Session State
        with st.expander("📌 Data yang Diupload", expanded=True):
            st.dataframe(df, width=800, height=400)

# 🔮 Tombol Prediksi
if "df" in st.session_state:
    df = st.session_state["df"]
    start_date = df.index[-1]
    start = st.sidebar.date_input("📅 Tanggal Mulai", value=start_date, disabled=True)
    end = st.sidebar.date_input("📅 Tanggal Selesai", value=None)
    button = st.sidebar.button("🔮 Prediksi!", type="primary")

    if button:
        if end is not None:
            if end > start:
                # Latih Model Saat Tombol Ditekan
                with st.spinner("🔄 Melatih Model..."):
                    model_fit, mae, rmse, mae_percentage,rmse_percentage = train_and_evaluate_model(df)

                if model_fit is not None:
                    evaluate_model(mae, rmse, mae_percentage,rmse_percentage)
                    predict(start, end, df, model_fit)
                else:
                    st.error("Model gagal dilatih. Pastikan data cukup.")
            else:
                st.warning("Tanggal selesai harus lebih besar dari tanggal mulai!")
        else:
            st.warning("Isi terlebih dahulu tanggal selesai!")
