import streamlit as st
from load_data import load_data
from result import result
# from datetime import date

st.title("Prediksi Kurs Jual Rupiah Terhadap Mata Uang Asing")
inform = st.info("Silahkan Masukan File Terlebih Dahulu...")
uploaded_files = st.file_uploader("Choose an Excel File", type=".xlsx")

if uploaded_files is not None:
    inform.empty()
    df = load_data(uploaded_files)
    start_date = df.index[-1]
    start = st.sidebar.date_input("Tanggal Mulai", value=start_date, disabled=True)
    end = st.sidebar.date_input("Tanggal Selesai", value=None)
    button = st.sidebar.button("Prediksi!", type="primary")
    st.sidebar.title("Data hasil Upload")
    st.sidebar.dataframe(df, width=800, height=400)
    
    if button:
        if start is not None and end is not None:
            result(start, end, df)
        else:
            st.info("Isi terlebih dahulu periodenya...")
    
