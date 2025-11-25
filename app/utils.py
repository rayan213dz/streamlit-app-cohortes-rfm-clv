import pandas as pd
import numpy as np


# ============================================================
# 1. Chargement & Nettoyage des données
# ============================================================

def load_data(path=None):
    """
    Charge le fichier nettoyé (CSV) généré par le notebook.
    - Convertit InvoiceDate en datetime
    - Renomme les colonnes pour compatibilité avec l'app
    - Calcule InvoiceMonth, Revenue, Year
    """

    # Lecture CSV

    default_url = "https://drive.google.com/uc?export=download&id=1j6MVlQrVzAh8xWkvNVtAfjCUw6sqe44R"

    if path is None:
        path = default_url
    
    df = pd.read_csv(path)

    # Normalisation des noms de colonnes
    df = df.rename(columns={
        "Is_Return": "IsReturn",
        "TotalAmount": "Revenue",
        "Customer ID": "Customer ID",  # on garde l'espace pour compatibilité
    })

    # Convertir les dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Calcul InvoiceMonth (important pour cohortes + visualisations)
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    # Année (utile pour filtres)
    df["Year"] = df["InvoiceDate"].dt.year

    # Revenue est déjà présent, mais on s’assure que c’est bien numeric
    df["Revenue"] = df["Revenue"].astype(float)

    # Assurer cohérence types
    df["Customer ID"] = df["Customer ID"].astype(int)

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



# ============================================================
# 6. CLV (formule fermée)
# ============================================================

def compute_clv_formula(r, d, m):
    """
    CLV = m * r / (1 + d - r)
    où :
    - r = taux de rétention mensuel
    - d = taux d'actualisation mensuel
    - m = marge moyenne par mois
    """

    if r >= 1:
        return np.inf

    return m * (r / (1 + d - r))


# ============================================================
# 7. Scénarios (simulation pour Streamlit)
# ============================================================

def simulate_scenarios(base_clv, retention, margin, retention_gain=0, margin_gain=0):
    """
    Retourne le CLV modifié après application :
        - d'un gain de rétention (%)
        - d'un gain de marge (%)
    """

    new_r = retention * (1 + retention_gain)
    new_m = margin * (1 + margin_gain)

    new_clv = compute_clv_formula(new_r, d=0.01, m=new_m)

    delta = new_clv - base_clv

    return new_clv, delta

def data_quality_report(df_raw, df_filtered):
    report = {}
    report["rows_total"] = len(df_raw)
    report["rows_filtered"] = len(df_filtered)
    report["pct_kept"] = len(df_filtered) / len(df_raw) * 100

    report["pct_missing_customer"] = df_raw["Customer ID"].isna().mean() * 100
    report["pct_returns"] = df_raw["Invoice"].astype(str).str.startswith("C").mean() * 100

    # part du CA liée aux retours (avant neutralisation)
    if "Revenue" in df_raw.columns:
        total_rev = df_raw["Revenue"].sum()
        returns_rev = df_raw.loc[df_raw["Revenue"] < 0, "Revenue"].sum()
        report["returns_share_revenue"] = returns_rev / total_rev * 100 if total_rev != 0 else 0
    return report


