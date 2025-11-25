import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats


# ============================================================
# 1. CHARGEMENT & NETTOYAGE DES DONNÉES
# ============================================================

def load_data(path="data/clean/online_retail_cleaned.csv"):
    """
    Charge le fichier nettoyé (CSV) généré par le notebook.
    - Convertit InvoiceDate en datetime
    - Renomme les colonnes pour compatibilité
    - Calcule InvoiceMonth, Revenue, Year
    """
    df = pd.read_csv(path)
    
    # Normalisation des noms de colonnes
    df = df.rename(columns={
        "Is_Return": "IsReturn",
        "TotalAmount": "Revenue",
        "Customer ID": "Customer ID",
    })
    
    # Convertir les dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Year"] = df["InvoiceDate"].dt.year
    df["Quarter"] = df["InvoiceDate"].dt.to_period("Q").dt.to_timestamp()
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Hour"] = df["InvoiceDate"].dt.hour
    
    # Revenue
    df["Revenue"] = df["Revenue"].astype(float)
    df["Customer ID"] = df["Customer ID"].astype(int)
    
    return df


# ============================================================
# 2. FILTRES STREAMLIT
# ============================================================

def apply_filters(df, start_date=None, end_date=None, countries=None, 
                  returns_mode="Inclure", min_order_value=0, customer_type=None):
    """
    Applique les filtres sélectionnés par l'utilisateur.
    
    Args:
        df: DataFrame source
        start_date: Date de début
        end_date: Date de fin
        countries: Liste des pays à inclure
        returns_mode: "Inclure", "Exclure", ou "Neutraliser"
        min_order_value: Seuil minimum de commande
        customer_type: Filtre par type de client (nouveau/récurrent)
    """
    df_filtered = df.copy()
    
    # Filtres temporels
    if start_date:
        df_filtered = df_filtered[df_filtered["InvoiceDate"] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered["InvoiceDate"] <= pd.to_datetime(end_date)]
    
    # Filtre pays
    if countries and len(countries) > 0:
        df_filtered = df_filtered[df_filtered["Country"].isin(countries)]
    
    # Gestion des retours
    if returns_mode == "Exclure":
        df_filtered = df_filtered[df_filtered["IsReturn"] == False]
    elif returns_mode == "Neutraliser":
        df_filtered.loc[df_filtered["IsReturn"], "Revenue"] = 0
    
    # Filtre valeur commande
    if min_order_value > 0:
        df_filtered = df_filtered[df_filtered["Revenue"].abs() >= min_order_value]
    
    return df_filtered


# ============================================================
# 3. ANALYSE COHORTES
# ============================================================

def compute_cohorts(df):
    """
    Calcule les métriques de cohortes d'acquisition.
    
    Returns:
        - cohort_counts: Nombre de clients uniques par cohorte et âge
        - retention_table: Table de rétention (%)
        - revenue_age: CA par âge de cohorte
        - ltv_by_cohort: LTV cumulée par cohorte
    """
    # Date de première commande par client
    first_purchase = df.groupby("Customer ID")["InvoiceDate"].min().rename("CohortStart")
    df = df.join(first_purchase, on="Customer ID")
    
    df["CohortMonth"] = df["CohortStart"].dt.to_period("M").dt.to_timestamp()
    df["InvoiceMonth"] = df["InvoiceMonth"].dt.to_period("M").dt.to_timestamp()
    
    # Âge de cohorte en mois
    df["CohortAge"] = (
        (df["InvoiceMonth"].dt.year - df["CohortMonth"].dt.year) * 12 +
        (df["InvoiceMonth"].dt.month - df["CohortMonth"].dt.month)
    )
    
    # Nombre de clients uniques par cohorte et âge
    cohort_counts = (
        df.groupby(["CohortMonth", "CohortAge"])["Customer ID"]
          .nunique()
          .unstack(fill_value=0)
    )
    
    # Table de rétention
    cohort_size = cohort_counts.iloc[:, 0]
    retention_table = cohort_counts.divide(cohort_size, axis=0)
    
    # CA par âge de cohorte
    revenue_age = (
        df.groupby(["CohortMonth", "CohortAge"])["Revenue"]
          .sum()
          .unstack(fill_value=0)
    )
    
    # LTV cumulée par cohorte
    ltv_by_cohort = revenue_age.cumsum(axis=1)
    
    return cohort_counts, retention_table, revenue_age, ltv_by_cohort


def compute_cohort_metrics(df):
    """
    Calcule des métriques avancées par cohorte :
    - Taux de conversion
    - Panier moyen par âge
    - Lifetime value
    - Churn rate
    """
    first_purchase = df.groupby("Customer ID")["InvoiceDate"].min().rename("CohortStart")
    df = df.join(first_purchase, on="Customer ID")
    
    df["CohortMonth"] = df["CohortStart"].dt.to_period("M").dt.to_timestamp()
    df["InvoiceMonth"] = df["InvoiceMonth"].dt.to_period("M").dt.to_timestamp()
    df["CohortAge"] = (
        (df["InvoiceMonth"].dt.year - df["CohortMonth"].dt.year) * 12 +
        (df["InvoiceMonth"].dt.month - df["CohortMonth"].dt.month)
    )
    
    # Panier moyen par cohorte et âge
    avg_basket = (
        df.groupby(["CohortMonth", "CohortAge"])["Revenue"]
          .mean()
          .unstack(fill_value=0)
    )
    
    # Fréquence d'achat par cohorte et âge
    purchase_freq = (
        df.groupby(["CohortMonth", "CohortAge"])["Invoice"]
          .nunique()
          .unstack(fill_value=0)
    )
    
    return avg_basket, purchase_freq


# ============================================================
# 4. SEGMENTATION RFM
# ============================================================

def compute_rfm(df):
    """
    Calcule la segmentation RFM avec scores et labels.
    
    Returns:
        DataFrame avec Recency, Frequency, Monetary, scores et segments
    """
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "Invoice": "nunique",
        "Revenue": "sum"
    })
    
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    
    # Scores RFM (1-5, quantiles)
    rfm["R_score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1], duplicates='drop')
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5], duplicates='drop')
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5], duplicates='drop')
    
    rfm["RFM_score"] = (
        rfm["R_score"].astype(int) +
        rfm["F_score"].astype(int) +
        rfm["M_score"].astype(int)
    )
    
    # Segmentation détaillée
    def label_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyaux"
        elif r >= 4 and f <= 2:
            return "Nouveaux"
        elif r <= 2 and f >= 3:
            return "À risque"
        elif r <= 2 and f <= 2 and m >= 3:
            return "Hibernants"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Perdus"
        else:
            return "Potentiels"
    
    rfm["Segment"] = rfm.apply(label_segment, axis=1)
    
    return rfm


def compute_rfm_trends(df, rfm):
    """
    Analyse l'évolution des segments RFM dans le temps.
    """
    # Calcul RFM sur différentes périodes (fenêtres glissantes)
    periods = df["InvoiceMonth"].unique()
    trends = []
    
    for period in sorted(periods)[-6:]:  # 6 derniers mois
        period_df = df[df["InvoiceMonth"] <= period]
        period_rfm = compute_rfm(period_df)
        segment_counts = period_rfm["Segment"].value_counts()
        segment_counts["Period"] = period
        trends.append(segment_counts)
    
    return pd.DataFrame(trends)


# ============================================================
# 5. CLV (CUSTOMER LIFETIME VALUE)
# ============================================================

def compute_clv_empirical(retention_table, revenue_age, horizon_months=12):
    """
    CLV empirique basée sur les cohortes historiques.
    
    Args:
        retention_table: Table de rétention
        revenue_age: CA par âge de cohorte
        horizon_months: Horizon temporel (en mois)
    
    Returns:
        CLV totale empirique
    """
    max_age = min(horizon_months, revenue_age.shape[1])
    clv = revenue_age.iloc[:, :max_age].mean().sum()
    return clv


def compute_clv_formula(r, d, m):
    """
    CLV via formule fermée : CLV = m × r / (1 + d - r)
    
    Args:
        r: Taux de rétention mensuel
        d: Taux d'actualisation mensuel
        m: Marge moyenne mensuelle par client
    
    Returns:
        CLV par client
    """
    if r >= 1 or (1 + d - r) <= 0:
        return np.inf
    
    return m * (r / (1 + d - r))


def compute_clv_probabilistic(df, rfm, discount_rate=0.01, periods=12):
    """
    Modèle probabiliste de CLV basé sur la probabilité d'achat.
    Utilise le modèle BG/NBD simplifié.
    
    Returns:
        DataFrame avec CLV probabiliste par client
    """
    # Calcul de la probabilité de survie client
    max_recency = df.groupby("Customer ID")["InvoiceDate"].max()
    snapshot = df["InvoiceDate"].max()
    days_since = (snapshot - max_recency).dt.days
    
    # Probabilité de survie (modèle exponentiel simplifié)
    lambda_param = 1 / rfm["Recency"].mean()
    survival_prob = np.exp(-lambda_param * days_since)
    
    # CLV = Valeur moyenne × Fréquence × Probabilité de survie × Facteur d'actualisation
    avg_order_value = rfm["Monetary"] / rfm["Frequency"]
    
    clv_prob = pd.DataFrame({
        "CustomerID": rfm.index,
        "AvgOrderValue": avg_order_value,
        "Frequency": rfm["Frequency"],
        "SurvivalProb": survival_prob.values,
        "CLV_Probabilistic": avg_order_value * rfm["Frequency"] * survival_prob.values
    })
    
    return clv_prob


# ============================================================
# 6. SIMULATION DE SCÉNARIOS
# ============================================================

def simulate_scenarios(base_clv, retention, margin, retention_gain=0, margin_gain=0, discount=0.01):
    """
    Simule l'impact de changements sur la CLV.
    
    Args:
        base_clv: CLV de référence
        retention: Taux de rétention actuel
        margin: Marge actuelle
        retention_gain: Variation du taux de rétention (ratio)
        margin_gain: Variation de la marge (ratio)
        discount: Taux d'actualisation
    
    Returns:
        new_clv: Nouvelle CLV
        delta_clv: Variation de CLV
        metrics: Dictionnaire de métriques détaillées
    """
    new_r = retention * (1 + retention_gain)
    new_r = min(new_r, 0.99)  # Cap à 99%
    
    new_m = margin * (1 + margin_gain)
    
    new_clv = compute_clv_formula(new_r, discount, new_m)
    delta_clv = new_clv - base_clv
    delta_pct = (delta_clv / base_clv * 100) if base_clv > 0 else 0
    
    metrics = {
        "base_clv": base_clv,
        "new_clv": new_clv,
        "delta_clv": delta_clv,
        "delta_pct": delta_pct,
        "new_retention": new_r,
        "new_margin": new_m,
        "retention_impact": (new_clv - compute_clv_formula(retention, discount, new_m)),
        "margin_impact": (new_clv - compute_clv_formula(new_r, discount, margin))
    }
    
    return new_clv, delta_clv, metrics


def sensitivity_analysis(retention, margin, discount=0.01, 
                        retention_range=(-20, 30), margin_range=(-30, 30), steps=20):
    """
    Analyse de sensibilité pour CLV.
    
    Returns:
        DataFrame avec CLV pour différentes combinaisons de paramètres
    """
    retention_values = np.linspace(
        retention * (1 + retention_range[0]/100),
        retention * (1 + retention_range[1]/100),
        steps
    )
    
    margin_values = np.linspace(
        margin * (1 + margin_range[0]/100),
        margin * (1 + margin_range[1]/100),
        steps
    )
    
    results = []
    for r in retention_values:
        for m in margin_values:
            clv = compute_clv_formula(min(r, 0.99), discount, m)
            results.append({
                "Retention": r,
                "Margin": m,
                "CLV": clv,
                "RetentionPct": (r - retention) / retention * 100,
                "MarginPct": (m - margin) / margin * 100
            })
    
    return pd.DataFrame(results)


# ============================================================
# 7. QUALITÉ DES DONNÉES
# ============================================================

def data_quality_report(df_raw, df_filtered):
    """
    Génère un rapport de qualité des données.
    
    Returns:
        Dictionnaire avec métriques de qualité
    """
    report = {}
    
    # Volumétrie
    report["rows_total"] = len(df_raw)
    report["rows_filtered"] = len(df_filtered)
    report["pct_kept"] = (len(df_filtered) / len(df_raw) * 100) if len(df_raw) > 0 else 0
    
    # Clients
    report["customers_total"] = df_raw["Customer ID"].nunique()
    report["customers_filtered"] = df_filtered["Customer ID"].nunique()
    
    # Valeurs manquantes
    report["pct_missing_customer"] = df_raw["Customer ID"].isna().mean() * 100
    report["pct_missing_description"] = df_raw.get("Description", pd.Series()).isna().mean() * 100
    
    # Retours
    report["pct_returns"] = (df_raw["Invoice"].astype(str).str.startswith("C").mean() * 100)
    
    # CA et retours
    total_revenue = df_raw["Revenue"].sum()
    returns_revenue = df_raw.loc[df_raw["Revenue"] < 0, "Revenue"].sum()
    report["total_revenue"] = total_revenue
    report["returns_revenue"] = returns_revenue
    report["returns_share_revenue"] = (abs(returns_revenue) / abs(total_revenue) * 100) if total_revenue != 0 else 0
    
    # Outliers (méthode IQR)
    Q1 = df_raw["Revenue"].quantile(0.25)
    Q3 = df_raw["Revenue"].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_raw["Revenue"] < (Q1 - 1.5 * IQR)) | (df_raw["Revenue"] > (Q3 + 1.5 * IQR)))
    report["pct_outliers"] = outliers.mean() * 100
    
    # Période couverte
    report["date_min"] = df_raw["InvoiceDate"].min()
    report["date_max"] = df_raw["InvoiceDate"].max()
    report["days_coverage"] = (report["date_max"] - report["date_min"]).days
    
    return report


# ============================================================
# 8. ANALYSES AVANCÉES
# ============================================================

def compute_seasonality(df):
    """
    Analyse la saisonnalité des ventes.
    
    Returns:
        DataFrame avec indicateurs de saisonnalité
    """
    # Par mois de l'année
    df["Month"] = df["InvoiceDate"].dt.month
    monthly_revenue = df.groupby("Month")["Revenue"].sum()
    
    # Par jour de la semaine
    df["DayName"] = df["InvoiceDate"].dt.day_name()
    daily_revenue = df.groupby("DayName")["Revenue"].sum()
    
    # Par heure
    hourly_revenue = df.groupby("Hour")["Revenue"].sum()
    
    return monthly_revenue, daily_revenue, hourly_revenue


def compute_customer_segments_value(df, rfm):
    """
    Calcule la valeur par segment RFM.
    
    Returns:
        DataFrame avec métriques par segment
    """
    # Jointure avec données transactionnelles
    df_with_segment = df.merge(
        rfm[["Segment"]], 
        left_on="Customer ID", 
        right_index=True, 
        how="left"
    )
    
    segment_metrics = df_with_segment.groupby("Segment").agg({
        "Customer ID": "nunique",
        "Invoice": "nunique",
        "Revenue": ["sum", "mean"],
        "Quantity": "sum"
    }).round(2)
    
    segment_metrics.columns = ["_".join(col).strip() for col in segment_metrics.columns]
    segment_metrics = segment_metrics.rename(columns={
        "Customer ID_nunique": "NbClients",
        "Invoice_nunique": "NbCommandes",
        "Revenue_sum": "CA_Total",
        "Revenue_mean": "PanierMoyen",
        "Quantity_sum": "QteTotale"
    })
    
    # CLV moyenne par segment
    segment_metrics["CA_Par_Client"] = segment_metrics["CA_Total"] / segment_metrics["NbClients"]
    
    return segment_metrics.reset_index()


def compute_churn_risk(rfm, threshold_days=90):
    """
    Identifie les clients à risque de churn.
    
    Args:
        rfm: DataFrame RFM
        threshold_days: Seuil de jours sans achat pour considérer un risque
    
    Returns:
        DataFrame avec score de risque
    """
    rfm_copy = rfm.copy()
    
    # Score de risque basé sur Recency
    rfm_copy["ChurnRisk"] = np.where(
        rfm_copy["Recency"] > threshold_days, "Élevé",
        np.where(rfm_copy["Recency"] > threshold_days/2, "Moyen", "Faible")
    )
    
    # Probabilité de churn (modèle simplifié)
    max_recency = rfm_copy["Recency"].max()
    rfm_copy["ChurnProbability"] = (rfm_copy["Recency"] / max_recency * 100).round(2)
    
    return rfm_copy


def compute_product_affinity(df):
    """
    Analyse les affinités produits (produits achetés ensemble).
    
    Returns:
        DataFrame avec paires de produits et fréquence
    """
    # Grouper par facture
    invoice_products = df.groupby("Invoice")["StockCode"].apply(list)
    
    # Recherche de paires (méthode simplifiée)
    from itertools import combinations
    
    pairs = []
    for products in invoice_products:
        if len(products) > 1:
            for pair in combinations(set(products), 2):
                pairs.append(sorted(pair))
    
    # Comptage
    pairs_df = pd.DataFrame(pairs, columns=["Product1", "Product2"])
    affinity = pairs_df.groupby(["Product1", "Product2"]).size().reset_index(name="Frequency")
    affinity = affinity.sort_values("Frequency", ascending=False)
    
    return affinity.head(50)


def compute_cac_metrics(df, marketing_spend=None):
    """
    Calcule le CAC (Customer Acquisition Cost) et métriques associées.
    
    Args:
        df: DataFrame des transactions
        marketing_spend: Budget marketing (optionnel)
    
    Returns:
        Dictionnaire de métriques CAC/LTV
    """
    # Nouveaux clients par mois
    first_purchase_dates = df.groupby("Customer ID")["InvoiceDate"].min()
    new_customers_by_month = first_purchase_dates.dt.to_period("M").value_counts().sort_index()
    
    metrics = {
        "total_new_customers": len(first_purchase_dates),
        "avg_new_customers_per_month": new_customers_by_month.mean()
    }
    
    # Si budget marketing fourni
    if marketing_spend:
        metrics["cac"] = marketing_spend / metrics["total_new_customers"]
        
        # LTV/CAC ratio (estimation)
        avg_revenue_per_customer = df.groupby("Customer ID")["Revenue"].sum().mean()
        metrics["ltv_cac_ratio"] = avg_revenue_per_customer / metrics["cac"]
        metrics["break_even_months"] = metrics["cac"] / (avg_revenue_per_customer / 12)
    
    return metrics


# ============================================================
# 9. PRÉDICTION & FORECASTING
# ============================================================

def forecast_revenue(df, periods=6):
    """
    Prévision simple du CA basée sur moyenne mobile.
    
    Args:
        df: DataFrame des transactions
        periods: Nombre de mois à prévoir
    
    Returns:
        DataFrame avec prévisions
    """
    # Agrégation mensuelle
    monthly_revenue = df.groupby("InvoiceMonth")["Revenue"].sum().sort_index()
    
    # Moyenne mobile sur 3 mois
    ma_3 = monthly_revenue.rolling(window=3).mean()
    
    # Tendance linéaire simple
    x = np.arange(len(monthly_revenue))
    y = monthly_revenue.values
    
    # Régression linéaire
    slope, intercept = np.polyfit(x, y, 1)
    
    # Prévisions
    last_index = len(monthly_revenue)
    future_x = np.arange(last_index, last_index + periods)
    forecast = slope * future_x + intercept
    
    # Création des dates futures
    last_date = monthly_revenue.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
    
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Revenue_Forecast": forecast,
        "Type": "Prévision"
    })
    
    # Historique
    history_df = pd.DataFrame({
        "Date": monthly_revenue.index,
        "Revenue_Forecast": monthly_revenue.values,
        "Type": "Historique"
    })
    
    return pd.concat([history_df, forecast_df], ignore_index=True)


# ============================================================
# 10. UTILITAIRES D'EXPORT
# ============================================================

def prepare_activation_list(rfm, segments_to_activate=None, top_n=None):
    """
    Prépare une liste activable pour les campagnes marketing.
    
    Args:
        rfm: DataFrame RFM
        segments_to_activate: Liste des segments à cibler
        top_n: Nombre de clients à extraire
    
    Returns:
        DataFrame prêt pour export
    """
    activation = rfm.copy()
    
    # Filtrer par segments si spécifié
    if segments_to_activate:
        activation = activation[activation["Segment"].isin(segments_to_activate)]
    
    # Trier par valeur
    activation = activation.sort_values("Monetary", ascending=False)
    
    # Limiter au top N
    if top_n:
        activation = activation.head(top_n)
    
    # Ajouter des colonnes utiles
    activation["DateExtraction"] = datetime.now().strftime("%Y-%m-%d")
    activation["Priorite"] = pd.cut(
        activation["Monetary"], 
        bins=3, 
        labels=["Basse", "Moyenne", "Haute"]
    )
    
    # Colonnes finales
    columns_export = [
        "Segment", "Recency", "Frequency", "Monetary",
        "RFM_score", "Priorite", "DateExtraction"
    ]
    
    return activation[columns_export].reset_index()


def generate_executive_summary(df, rfm, cohort_counts, retention_table):
    """
    Génère un résumé exécutif avec les KPIs principaux.
    
    Returns:
        Dictionnaire avec métriques clés
    """
    summary = {}
    
    # Clients & Transactions
    summary["total_customers"] = df["Customer ID"].nunique()
    summary["total_transactions"] = df["Invoice"].nunique()
    summary["total_revenue"] = df["Revenue"].sum()
    
    # Panier moyen
    summary["avg_basket"] = df.groupby("Invoice")["Revenue"].sum().mean()
    
    # Fréquence d'achat moyenne
    summary["avg_purchase_frequency"] = df.groupby("Customer ID")["Invoice"].nunique().mean()
    
    # Rétention moyenne (M+1)
    if retention_table.shape[1] > 1:
        summary["avg_retention_m1"] = retention_table.iloc[:, 1].mean()
    else:
        summary["avg_retention_m1"] = 0
    
    # Distribution RFM
    summary["rfm_distribution"] = rfm["Segment"].value_counts().to_dict()
    
    # Top segment par CA
    segment_revenue = df.merge(rfm[["Segment"]], left_on="Customer ID", right_index=True)
    top_segment = segment_revenue.groupby("Segment")["Revenue"].sum().idxmax()
    summary["top_segment"] = top_segment
    
    # Période analysée
    summary["date_start"] = df["InvoiceDate"].min()
    summary["date_end"] = df["InvoiceDate"].max()
    summary["analysis_days"] = (summary["date_end"] - summary["date_start"]).days
    
    return summary
