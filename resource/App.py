import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="App-Predict", layout="wide", page_icon="💰")

# 📌 Load Data dari File Excel
def load_data(uploaded_file):
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
        
        if len(df) < 1000:
            st.error("Data terlalu sedikit untuk diproses. Pastikan minimal ada 1000 data.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {e}")
        return None

# 🔥 Fungsi Train Model dengan Train-Test Split dan Walk-Forward Validation
def train_and_evaluate_model(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    model = ARIMA(train['Kurs Jual'], order=(2, 1, 2))
    model_fit = model.fit()

    history = list(train['Kurs Jual'])
    predictions = []
    actual_values = list(test['Kurs Jual'])

    for t in range(len(test)):
        model = ARIMA(history, order=(2, 1, 2))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(actual_values[t])

    mae = float(mean_absolute_error(actual_values, predictions))
    mean_actual = np.mean(actual_values)
    mae_percentage = (mae / mean_actual) * 100

    return model_fit, mae_percentage

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
def evaluate_model(mae_percentage):
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        st.markdown(f"""
        <style>
            .metric-card {{
                background: #FBF8EF;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
                text-align: center;
                width: 250px;
                margin: auto;
                transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            }}

            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 6px 6px 12px rgba(0, 0, 0, 0.3);
            }}

            .metric-title {{
                font-size: 18px;
                font-weight: bold;
                color: black;
                margin-bottom: 5px;
            }}

            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #FFB433;
            }}
        </style>

        <div class="metric-card">
            <div class="metric-title">MAE (Mean Absolute Error)</div>
            <div class="metric-value">{mae_percentage:.4f}%</div>
        </div>
    """, unsafe_allow_html=True)


# 💹 Tampilan Utama
st.markdown("""
    <style>
        .card-container {
            display: flex;
            justify-content: center;
        }

        .card {
            background: #FBF8EF;
            padding: 20px;
            width: 60%; 
            max-width: 1000px; 
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            text-align: center;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }

        .title {
            font-size: 28px; 
            font-weight: bold;
            margin: 0;
        }

        .highlight {
            color: #FFB433;
        }

        @media (max-width: 600px) {
            .card {
                width: 90%; 
            }
            .title {
                font-size: 24px; 
            }
        }
    </style>

    <div class="card-container">
        <div class="card">
            <h1 class="title">
                Bersiap untuk Masa Depan! <br> 
                <span class="highlight">Lihat Tren Kurs Rupiah</span> dengan Model Prediksi Canggih! 🚀
            </h1>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            background-color: #80CBC4;
            box-shadow: 4px 0 10px rgba(0, 0, 0, 0.2);
            border-right: 5px solid #B4EBE6 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
        @keyframes bounce {
            0% { transform: translateY(-5px); opacity: 1; }
            50% { transform: translateY(5px); opacity: 1; }
            100% { transform: translateY(0); opacity: 1; }
        }

        div[data-testid="stAlert"] {
            background-color: #B4EBE6 !important;
            border-left: 5px solid #FFB433 !important;
            border-radius: 10px;
            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
            animation: bounce 1.5s ease-in-out infinite alternate;
        }

        div[data-testid="stAlert"] p {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)


# 📂 Upload File
uploaded_file = st.sidebar.file_uploader("📂 Pilih File Excel", type=".xlsx", help="Pastikan file yang diunggah dalam format .xlsx dengan data kurs jual rupiah.")

if uploaded_file is None:
    st.info("**Siapkan data historismu terlebih dahulu...** \n Kamu bisa mengambilnya dari situs resmi Bank Indonesia: [Kurs BI](https://www.bi.go.id/id/statistik/informasi-kurs/transaksi-bi/default.aspx)", icon="💡")
    st.stop()

df = load_data(uploaded_file)

if df is None or df.empty:
    st.stop()

with st.sidebar.expander("📌 Data yang Diupload", expanded=True):
    st.sidebar.dataframe(df, width=900, height=400)


today = pd.Timestamp.today().date()
last_data_date = df.index[-1].date()

if today > last_data_date:
    start_date = today  
else:
    start_date = last_data_date 
# 📅 Input Tanggal untuk Prediksi
start = st.sidebar.date_input("📅 Tanggal Mulai", value=start_date, disabled=True)
end = st.sidebar.date_input("📅 Tanggal Selesai", value=None, min_value=start_date + pd.Timedelta(days=1))

if st.sidebar.button("🔮 Prediksi!", type="primary", use_container_width=True):
    if end is None:
        st.warning("Silakan pilih tanggal selesai terlebih dahulu.")
    elif end <= start:
        st.warning("Tanggal selesai harus lebih besar dari tanggal mulai!")
    else:
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
            <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            @keyframes pulse {
                0% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1); opacity: 0.6; }
                100% { transform: scale(1); opacity: 1; }
            }

            .loading-container {
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                font-weight: bold;
                color: #ffffff;
                background-color: #ff4b4b;
                padding: 10px 20px;
                border-radius: 10px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
                width: 300px;
                margin: auto;
            }

            .loading-icon {
                display: inline-block;
                margin-right: 10px;
                width: 20px;
                height: 20px;
                border: 3px solid white;
                border-top: 3px solid transparent;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            .loading-text {
                display: inline-block;
                animation: pulse 2s ease-in-out infinite;
            }
            </style>

            <div class="loading-container">
                <div class="loading-icon"></div>
                <div class="loading-text">Tunggu Sebentar...</div>
            </div>
        """, unsafe_allow_html=True)

        model_fit, mae_percentage = train_and_evaluate_model(df)
        
        if model_fit is not None:
            loading_placeholder.empty()
            evaluate_model(mae_percentage)
            predict(start, end, df, model_fit)
        else:
            st.error("Model gagal dilatih. Pastikan data cukup.")
