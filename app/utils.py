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
