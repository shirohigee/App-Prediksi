import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ✅ Page config
st.set_page_config(page_title="App-Predict", layout="wide", page_icon="💰")

# 🧼 Load dan Preprocessing Data
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        df = df.iloc[3:].reset_index(drop=True)
        df.columns = ["NO", "Nilai", "Kurs Jual", "Kurs Beli", "Tanggal"]
        df = df[["Tanggal", "Kurs Jual"]]
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%m/%d/%Y %I:%M:%S %p", errors='coerce')
        df = df.dropna(subset=["Tanggal"])
        df["Kurs Jual"] = df["Kurs Jual"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        df["Kurs Jual"] = pd.to_numeric(df["Kurs Jual"], errors='coerce')
        df = df.dropna(subset=["Kurs Jual"])
        df.set_index("Tanggal", inplace=True)
        df.sort_index(ascending=True, inplace=True)
        df = df.asfreq('D')
        df["Kurs Jual"] = df["Kurs Jual"].interpolate(method='linear')

        if len(df) < 1000:
            st.warning("Data terlalu sedikit untuk diproses. Pastikan minimal ada 1000 data.")
            return None
        return df
    except Exception:
        st.warning(f"Data harus sesuai format Bank Indonesia...")
        return None

# 🧠 Training Model ARIMA
@st.cache_resource
def train_and_evaluate_model(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    model = ARIMA(train['Kurs Jual'], order=(2, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    history = list(train['Kurs Jual'])
    predictions = []
    actual_values = list(test['Kurs Jual'])

    for t in range(len(test)):
        model = ARIMA(history, order=(2, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(actual_values[t])

    mae = float(mean_absolute_error(actual_values, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual_values, predictions)))
    # mean_actual = np.mean(actual_values)
    mape_percentage = np.mean(
        np.abs(np.array(actual_values) - np.array(predictions)) / np.array(actual_values)
    ) * 100

    return model_fit, mae, rmse, mape_percentage

# 🔮 Prediksi Masa Depan
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
    fig.update_layout(title="Tren Kurs Rupiah", xaxis_title="Tanggal", yaxis_title="Kurs Jual", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# 📊 Evaluasi Model
def evaluate_model(mae, rmse, mape_percentage):
    col1, col2, col3 = st.columns([1, 1, 1])

    st.markdown("""
        <style>
            .metric-card {
                background: #FEF3E2;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2);
                text-align: center;
                width: 250px;
                margin: auto;
                transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            }

            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 6px 6px 12px rgba(0, 0, 0, 0.3);
            }

            .metric-title {
                font-size: 18px;
                font-weight: bold;
                color: black;
                margin-bottom: 5px;
            }

            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #FA4032;
            }

            @media (prefers-color-scheme: dark) {
                .metric-card { background: #2d2d2d; color: #f1f1f1; }
                .metric-title { color: #f1f1f1; }
                .metric-value { color: #FFA07A; }
            }
        </style>
    """, unsafe_allow_html=True)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAE (Mean Absolute Error)</div>
            <div class="metric-value">{mae:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE (Root Mean Squared Error)</div>
            <div class="metric-value">{rmse:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAPE (Mean Absolute Percentage Error)</div>
            <div class="metric-value">{mape_percentage:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# 🌈 Tampilan dan Style
st.markdown("""
    <style>
        .card-container {
            display: flex;
            justify-content: center;
        }

        .card {
            background: #FEF3E2;
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
            margin: 0;
        }

        .highlight {
            color: #FA4032;
        }

        section[data-testid="stSidebar"] {
            background-color: #FAB12F;
            box-shadow: 4px 0 10px rgba(0, 0, 0, 0.2);
            border-right: 1px solid #FA812F;
        }

        @keyframes bounce {
            0% { transform: translateY(0); opacity: 1; }
            50% { transform: translateY(-2px); opacity: 1; }
            100% { transform: translateY(0); opacity: 1; }
        }

        div[data-testid="stAlert"] {
            background-color: #FEF3E2;
            border-left: 5px solid #FAB12F;
            border-radius: 10px;
            box-shadow: 4px 4px 4px rgba(0, 0, 0, 0.2);
            animation: bounce 1s ease-in-out infinite alternate;
        }

        div[data-testid="stAlert"] p {
            color: black;
        }

        @media (prefers-color-scheme: dark) {
            .card { background: #1e1e1e; color: #f1f1f1; }
            .highlight { color: #FF7B72; }
            section[data-testid="stSidebar"] { background-color: #2d2d2d; border-right: 1px solid #444; }
            div[data-testid="stAlert"] { background-color: #2d2d2d; border-left: 5px solid #FFB347; }
            div[data-testid="stAlert"] p { color: #f1f1f1; }
            .element-container iframe { filter: invert(1) hue-rotate(180deg); }
        }
    </style>
""", unsafe_allow_html=True)

# 🎯 Hero Section
st.markdown("""
    <div class="card-container">
        <div class="card">
            <h1 class="title">
                Bersiap untuk Masa Depan! <br>
                <span class="highlight">Lihat Tren Kurs Rupiah</span> dengan Model Prediksi Canggih! 🚀
            </h1>
        </div>
    </div>
""", unsafe_allow_html=True)

# 📂 Upload Data
uploaded_file = st.file_uploader("📂 Upload File Excel", type=".xlsx", help="Pastikan file dalam format kurs jual dari BI.")

if uploaded_file is None:
    st.warning("Siapkan data historismu terlebih dahulu! Kamu bisa mengambilnya dari situs resmi Bank Indonesia: [Kurs BI](https://www.bi.go.id/id/statistik/informasi-kurs/transaksi-bi/default.aspx)", icon="💡")
    st.stop()

df = load_data(uploaded_file)
if df is None or df.empty:
    st.stop()

with st.sidebar:
    st.subheader("📌 Data yang Diupload")
    st.dataframe(df, width=2000, height=400)

today = pd.Timestamp.today().date()
last_data_date = df.index[-1].date()
start_date = today if today > last_data_date else last_data_date

st.sidebar.subheader("📅 Input Tanggal untuk Prediksi")
start = st.sidebar.date_input("Tanggal Mulai", value=start_date, disabled=True)
end = st.sidebar.date_input("**Tanggal Selesai**", value=None, min_value=start_date + pd.Timedelta(days=1))

if st.sidebar.button("🔮 Prediksi!", type="primary", use_container_width=True):
    if end is None:
        st.warning("Silakan pilih tanggal selesai terlebih dahulu.")
    elif end <= start:
        st.warning("Tanggal selesai harus lebih besar dari tanggal mulai!")
    else:
        model_fit, mae, rmse, mape_percentage = train_and_evaluate_model(df)
        if model_fit is not None:
            evaluate_model(mae, rmse, mape_percentage)
            predict(start, end, df, model_fit)
        else:
            st.warning("Model gagal dilatih. Pastikan data cukup.")
