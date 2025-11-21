import pandas as pd
import numpy as np


# ============================================================
# 1. Chargement & Nettoyage des données
# ============================================================

def load_data(path="data/raw/online_retail_II.xlsx"):
    """
    Charge et prépare les données Online Retail II (2009–2011).
    - Fusionne les 2 feuilles Excel
    - Nettoie les CustomerID
    - Convertit les prix et dates
    - Ajoute Revenue, IsReturn, InvoiceMonth, Year
    """

    df_2009 = pd.read_excel(path, sheet_name="Year 2009-2010")
    df_2010 = pd.read_excel(path, sheet_name="Year 2010-2011")
    df = pd.concat([df_2009, df_2010], ignore_index=True)

    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)

    df["IsReturn"] = df["Invoice"].astype(str).str.startswith("C")
    df["Revenue"] = df["Quantity"] * df["Price"]

    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df["InvoiceDate"].dt.year

    return df

# ============================================================
# 2. Filtres Streamlit
# ============================================================

def apply_filters(df, start_date=None, end_date=None, countries=None, returns_mode="Inclure"):

    if start_date:
        df = df[df["InvoiceDate"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["InvoiceDate"] <= pd.to_datetime(end_date)]

    if countries:
        df = df[df["Country"].isin(countries)]

    if returns_mode == "Exclure":
        df = df[df["IsReturn"] == False]

    if returns_mode == "Neutraliser":
        df = df.copy()
        df.loc[df["IsReturn"], "Revenue"] = 0

    return df
