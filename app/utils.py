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

# ============================================================
# 3. Cohortes
# ============================================================

def compute_cohorts(df):
    """
    Retourne :
        - cohort_counts : table des clients uniques (Cohorte × Âge)
        - retention : table en %
        - revenue_age : CA par âge de cohorte
    """

    # 1) Date de première commande
    first_purchase = df.groupby("Customer ID")["InvoiceDate"].min().rename("CohortStart")
    df = df.join(first_purchase, on="Customer ID")

    df["CohortMonth"] = df["CohortStart"].dt.to_period("M").dt.to_timestamp()
    df["InvoiceMonth"] = df["InvoiceMonth"].dt.to_period("M").dt.to_timestamp()

    # 2) Age de cohorte
    df["CohortAge"] = (
        (df["InvoiceMonth"].dt.year - df["CohortMonth"].dt.year) * 12 +
        (df["InvoiceMonth"].dt.month - df["CohortMonth"].dt.month)
    )

    # 3) Nombre de clients uniques
    cohort_counts = (
        df.groupby(["CohortMonth", "CohortAge"])["Customer ID"]
          .nunique()
          .unstack(fill_value=0)
    )

    # 4) Rétention
    cohort_size = cohort_counts.iloc[:, 0]
    retention = cohort_counts.divide(cohort_size, axis=0)

    # 5) CA par âge de cohorte
    revenue_age = (
        df.groupby(["CohortMonth", "CohortAge"])["Revenue"]
          .sum()
          .unstack(fill_value=0)
    )

    return cohort_counts, retention, revenue_age

# ============================================================
# 4. RFM
# ============================================================

def compute_rfm(df):
    """
    Calcule Recency, Frequency, Monetary
    Ajoute :
        - scores R, F, M (1–5)
        - score global
        - label segment
    """

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "Revenue": "sum"
    })

    rfm.columns = ["Recency", "Frequency", "Monetary"]

    # Scores RFM (quantiles)
    rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5])

    rfm["RFM_score"] = (
        rfm["R_score"].astype(int) +
        rfm["F_score"].astype(int) +
        rfm["M_score"].astype(int)
    )

    # Segments
    def label_segment(score):
        if score >= 13:
            return "Champions"
        if score >= 10:
            return "Loyaux"
        if score >= 7:
            return "Potentiels"
        if score >= 5:
            return "À risque"
        return "Perdus"

    rfm["Segment"] = rfm["RFM_score"].apply(label_segment)

    return rfm

# ============================================================
# 5. CLV (empirique via cohortes)
# ============================================================

def compute_clv_empirical(retention_table, revenue_age, horizon_months=12):
    """
    CLV empirique = somme du CA moyen par âge de cohorte sur un horizon donné
    """

    clv = revenue_age.iloc[:, :horizon_months].mean().sum()
    return clv