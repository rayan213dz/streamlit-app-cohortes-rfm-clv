import streamlit as st
import pandas as pd
import os

# ----------------------------------------
# Chargement des données
# ----------------------------------------

@st.cache_data
def load_data():
    base_path = os.path.join("data", "raw")

    files = ["Cohortes.csv", "Cohortes2.csv"]
    dfs = []

    for file in files:
        path = os.path.join(base_path, file)
        if os.path.exists(path):
            df = pd.read_csv(
                path,
                sep=";",
                decimal=",",
                encoding="utf-8"
            )
            dfs.append(df)
        else:
            st.warning(f"⚠️ Fichier introuvable : {file}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# ----------------------------------------
# APP STREAMLIT
# ----------------------------------------

st.title("V1 - Analyse Cohortes / RFM / CLV (Expérimentation)")

st.write("""
Bienvenue dans la **version d'expérimentation**.
Cette V1 permet de tester le chargement des données et un premier aperçu visuel.
""")

# Chargement
df = load_data()

if df.empty:
    st.error("❌ Aucune donnée chargée. Vérifie que Cohortes.csv & Cohortes2.csv sont dans data/raw/")
    st.stop()

# Nettoyage
df = df.rename(columns={
    "Invoice": "InvoiceNo",
    "Price": "UnitPrice",
    "Customer ID": "CustomerID"
})

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

df["LineAmount"] = df["Quantity"] * df["UnitPrice"]

# ----------------------------------------
# AFFICHAGE
# ----------------------------------------

st.subheader("Aperçu des données")
st.dataframe(df.head())

st.subheader("Statistiques globales")

col1, col2, col3 = st.columns(3)
col1.metric("Nombre de ventes", len(df))
col2.metric("Produits uniques", df["StockCode"].nunique())
col3.metric("Clients uniques", df["CustomerID"].nunique())

st.subheader("CA par pays")
df_ca = df.groupby("Country")["LineAmount"].sum().sort_values(ascending=False)

st.bar_chart(df_ca)
