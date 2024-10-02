import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def result(start, end, df):
    selisih_hari = (end - start).days + 1

    model = ARIMA(df["Kurs Jual"], order=(1,0,1))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=selisih_hari)
    forecast = forecast.round(2)

    last_date = df.index[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, selisih_hari + 1)]
    forecast_df = pd.DataFrame({"Tanggal": forecast_dates, "Prediksi Kurs Jual": forecast})
    forecast_df.set_index("Tanggal", inplace=True)
    
    # fitur yang ditambahkan
    formatted_forecast_df = forecast_df.copy()
    formatted_forecast_df["Prediksi Kurs Jual (Rp.)"] = formatted_forecast_df["Prediksi Kurs Jual"].apply(lambda x: f"Rp. {x:,.0f}")
    
    st.subheader(f"Data Frame - Data Prediksi {selisih_hari} Hari Kedepan")
    st.dataframe(formatted_forecast_df["Prediksi Kurs Jual (Rp.)"], width=600)
    
    st.bar_chart(forecast_df["Prediksi Kurs Jual"])