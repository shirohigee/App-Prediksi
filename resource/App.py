import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Fungsi untuk Load Data ===
def load_data(uploaded_file):
    if uploaded_file is None:
        return None

    try:
        df = pd.read_excel(uploaded_file)
        
        # Mengasumsikan header tambahan, mulai membaca dari baris ke-4
        df = df.iloc[3:].reset_index(drop=True)
        df.columns = ["NO", "Nilai", "Kurs Jual", "Kurs Beli", "Tanggal"]
        
        # Hanya ambil kolom yang dibutuhkan
        df = df[["Tanggal", "Kurs Jual"]]
        
        # Konversi tanggal
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors='coerce')
        df = df.dropna(subset=["Tanggal"])  # Hapus baris dengan tanggal tidak valid
        
        # Bersihkan data "Kurs Jual" dari karakter non-angka
        df["Kurs Jual"] = df["Kurs Jual"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df["Kurs Jual"] = pd.to_numeric(df["Kurs Jual"], errors='coerce')
        df = df.dropna(subset=["Kurs Jual"])  # Hapus baris yang masih kosong setelah pembersihan
        
        # Set indeks sebagai tanggal dan urutkan
        df.set_index("Tanggal", inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        return None

# === Fungsi untuk Prediksi ARIMA ===
def result(start, end, df):
    df.index = pd.to_datetime(df.index)
    selisih_hari = (end - start).days + 1

    if len(df) < 10:
        st.error("Data terlalu sedikit untuk membangun model ARIMA. Pastikan minimal ada 10 data.")
        return

    model = ARIMA(df["Kurs Jual"], order=(7,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=selisih_hari).round(2)
    
    last_date = df.index[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, selisih_hari + 1)]
    forecast_df = pd.DataFrame({"Tanggal": forecast_dates, "Prediksi Kurs Jual": forecast})
    forecast_df.set_index("Tanggal", inplace=True)
    forecast_df["Prediksi Kurs Jual (Rp.)"] = forecast_df["Prediksi Kurs Jual"].apply(lambda x: f"Rp. {x:,.0f}")
    
    st.subheader(f"Data Prediksi {selisih_hari} Hari Kedepan")
    st.dataframe(forecast_df[["Prediksi Kurs Jual (Rp.)"]], width=600)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df.index, df["Kurs Jual"], label="Data Historis", marker='o',markersize=1, linestyle='-')
    ax.plot(forecast_df.index, forecast_df["Prediksi Kurs Jual"], label="Prediksi", marker='o',markersize=1, linestyle='dashed', color='red')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Prediksi Kurs Jual dengan ARIMA")
    ax.set_ylabel("Kurs Jual")
    ax.legend()
    plt.xticks(rotation=45)
    
    st.pyplot(fig)

    # === Evaluasi Model ===
    train_size = int(len(df) * 0.8)  # 80% data untuk training
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    model_train = ARIMA(train["Kurs Jual"], order=(7,1,2))
    model_train_fit = model_train.fit()
    test_forecast = model_train_fit.forecast(steps=len(test))
    
    mae = mean_absolute_error(test, test_forecast)
    mse = mean_squared_error(test, test_forecast)
    rmse = np.sqrt(mse)
    
    # Konversi ke persentase dari nilai rata-rata
    mean_actual = test.mean()
    mae_pct = (mae / mean_actual) * 100
    mse_pct = (mse / mean_actual) * 100
    rmse_pct = (rmse / mean_actual) * 100
    
    st.subheader("Evaluasi Model ARIMA")
    st.write(f"**Mean Absolute Error (MAE):** {mae_pct.iloc[0]:.2f}%")
    st.write(f"**Mean Squared Error (MSE):** {mse_pct.iloc[0]:.2f}%")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse_pct.iloc[0]:.2f}%")

# === Streamlit App ===
st.title("Prediksi Kurs Jual Rupiah Terhadap Mata Uang Asing")
inform = st.info("Silahkan Masukkan File Terlebih Dahulu...")
uploaded_files = st.file_uploader("Pilih File Excel", type=".xlsx")

if uploaded_files is not None:
    inform.empty()
    df = load_data(uploaded_files)

    if df is None or df.empty:
        st.error("File tidak valid atau tidak mengandung data yang diperlukan.")
    else:
        with st.sidebar.expander("\U0001F4C2 Data yang Diupload", expanded=True):
            st.dataframe(df, width=800, height=400)
        
        start_date = df.index[-1] if not df.empty else None
        start = st.sidebar.date_input("Tanggal Mulai", value=start_date, disabled=True)
        end = st.sidebar.date_input("Tanggal Selesai", value=None)
        start = pd.to_datetime(start)
        end = pd.to_datetime(end) if end else None
        
        button = st.sidebar.button("\U0001F52E Prediksi!", type="primary")

        if button:
            if end is not None:
                if end > start:
                    result(start, end, df)
                else:
                    st.warning("Tanggal selesai harus lebih besar dari tanggal mulai!")
            else:
                st.warning("Isi terlebih dahulu tanggal selesai!")
