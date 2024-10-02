import pandas as pd
def load_data(uploaded_files):
    df = pd.read_excel(uploaded_files, skiprows=3)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header

    use_columns = ["Kurs Jual", "Tanggal"]
    df = df[use_columns]

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%m/%d/%Y %I:%M:%S %p")
    df["Kurs Jual"] = df["Kurs Jual"].astype(str).str.split(".").str[0].astype(int)

    df.set_index("Tanggal", inplace=True)
    df.sort_index(ascending=True, inplace=True)
    return df