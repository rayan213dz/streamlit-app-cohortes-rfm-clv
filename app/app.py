import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data,
    apply_filters,
    compute_cohorts,
    compute_rfm,
    compute_clv_empirical,
    compute_clv_formula,
    simulate_scenarios,
    data_quality_report
)

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Marketing Cohortes & CLV",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Application Marketing : Cohortes, RFM & CLV")

st.caption(
    "Ordre de lecture recommandé : *KPIs → Cohortes → Segments → Scénarios → Export*"
)


# ============================================================
# 1. CHARGEMENT & CACHE DES DONNÉES
# ============================================================

@st.cache_data
def load_raw_data():
    df = load_data("data/raw/online_retail_II.xlsx")
    return df


df_raw = load_raw_data()


# ============================================================
# 2. FILTRES GLOBAUX
# ============================================================

st.sidebar.header("🔍 Filtres globaux")

min_date = df_raw["InvoiceDate"].min().date()
max_date = df_raw["InvoiceDate"].max().date()

start_date = st.sidebar.date_input("Date de début", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Date de fin", max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("La date de début doit être ≤ date de fin.")

countries = st.sidebar.multiselect(
    "Pays",
    options=sorted(df_raw["Country"].unique()),
    default=["United Kingdom"] if "United Kingdom" in df_raw["Country"].unique() else [],
)

returns_mode = st.sidebar.radio(
    "Gestion des retours",
    ["Inclure", "Exclure", "Neutraliser"],
    help=(
        "Inclure : les retours sont considérés comme des ventes négatives.\n"
        "Exclure : les retours sont supprimés.\n"
        "Neutraliser : le CA des retours est mis à 0."
    ),
)

min_order_value = st.sidebar.number_input(
    "Seuil minimum de commande (Revenue)",
    min_value=0.0,
    value=0.0,
    step=10.0,
)

# application filtres
df = apply_filters(
    df_raw,
    start_date=start_date,
    end_date=end_date,
    countries=countries if len(countries) > 0 else None,
    returns_mode=returns_mode,
)

if min_order_value > 0:
    df = df[df["Revenue"].abs() >= min_order_value]

# Badge retours
filters_badge = f"Période : {start_date} → {end_date} | n transactions = {len(df):,}"
st.markdown(f"*Filtres actifs :* {filters_badge}")

if returns_mode == "Exclure":
    st.markdown("🟧 *Retours exclus*")
elif returns_mode == "Neutraliser":
    st.markdown("🟦 *Retours neutralisés (CA = 0)*")

st.markdown("---")


# ============================================================
# 3. PRÉ-CALCULS COMMUNS (COHORTES, RFM, CLV)
# ============================================================

if df.empty:
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()

cohort_counts, retention_table, revenue_age = compute_cohorts(df)
rfm = compute_rfm(df)

# CLV empirique : on prend un horizon de 12 mois
clv_emp_total = compute_clv_empirical(retention_table, revenue_age, horizon_months=12)
n_customers = df["Customer ID"].nunique()
clv_emp_per_cust = clv_emp_total / max(n_customers, 1)

# CLV formule fermée :
# r ~ rétention moyenne M+1 (taux de clients encore actifs à 1 mois)
if retention_table.shape[1] > 1:
    r_hat = retention_table.iloc[:, 1].mean()
else:
    r_hat = retention_table.iloc[:, 0].mean()

# marge moyenne mensuelle : on suppose marge = 30% du CA moyen mensuel par client
horizon_months = (df["InvoiceMonth"].max().to_period("M") - df["InvoiceMonth"].min().to_period("M")).n + 1
total_revenue = df["Revenue"].sum()
avg_monthly_rev_per_cust = total_revenue / max(horizon_months * n_customers, 1)
margin_rate_default = 0.30
m_margin = avg_monthly_rev_per_cust * margin_rate_default

d_discount = 0.01  # 1% de taux d'actualisation mensuel
clv_formula_per_cust = compute_clv_formula(r=r_hat, d=d_discount, m=m_margin)


# ============================================================
# 4. NAVIGATION ENTRE LES PAGES
# ============================================================

page = st.sidebar.radio(
    "📂 Navigation",
    ["KPIs (Overview)", "Cohortes (Diagnostiquer)", "Segments RFM (Prioriser)", "Scénarios (Simuler)", "Plan d’action & Export","Data Quality & Coverage"],
)


# ============================================================
# PAGE 1 : KPIs (OVERVIEW)
# ============================================================
if page == "KPIs (Overview)":
    st.subheader("📌 KPIs principaux")

    col1, col2, col3, col4, col5 = st.columns(5)

    # KPI 1 : Clients actifs
    active_customers = n_customers
    with col1:
        st.metric("Clients uniques (n)", value=f"{active_customers:,}")
        with st.expander("ℹ Clients uniques"):
            st.write(
                "Nombre de *clients distincts* ayant passé au moins une commande sur "
                "la période filtrée. Exemple : si 3 clients A, B, C ont commandé, alors n = 3."
            )

    # KPI 2 : CA total
    with col2:
        st.metric("CA total filtré", value=f"{total_revenue:,.0f} £")
        with st.expander("ℹ Chiffre d'affaires (CA)"):
            st.write(
                "Somme du *Revenue* sur la période filtrée. "
                "Revenue = Quantity × Price (les retours peuvent être négatifs ou neutralisés)."
            )

    # KPI 3 : CA moyen à 90 jours par nouveau client (approximation via CLV empirique 3 mois)
    clv_3m = compute_clv_empirical(retention_table, revenue_age, horizon_months=min(3, revenue_age.shape[1]))
    clv_3m_per_cust = clv_3m / max(n_customers, 1)

    with col3:
        st.metric("CA 90j moyen / client", value=f"{clv_3m_per_cust:,.2f} £")
        with st.expander("ℹ CA à 90 jours par client"):
            st.write(
                "Somme du CA moyen par âge de cohorte sur les *3 premiers mois* "
                "divisée par le nombre de clients. Illustration :\n\n"
                "- Mois 0 : 20£, Mois 1 : 10£, Mois 2 : 5£ ⇒ CLV_90j = 35£."
            )

    # KPI 4 : CLV empirique (12 mois)
    with col4:
        st.metric("CLV empirique 12 mois / client", value=f"{clv_emp_per_cust:,.2f} £")
        with st.expander("ℹ CLV empirique"):
            st.write(
                "CLV empirique = somme du *CA moyen par âge de cohorte* sur un horizon donné "
                "(ici 12 mois), *divisée par le nombre de clients*.\n\n"
                "On observe ce que les cohortes passées ont réellement dépensé."
            )

    # KPI 5 : CLV (formule fermée)
    with col5:
        st.metric("CLV formule / client", value=f"{clv_formula_per_cust:,.2f} £")
        with st.expander("ℹ CLV formule fermée"):
            st.write(
                "Formule : *CLV = m × r / (1 + d − r)*\n\n"
                "- r : taux de rétention mensuel moyen (ici ≈ rétention M+1)\n"
                "- d : taux d'actualisation mensuel (ici 1%)\n"
                "- m : marge moyenne par mois et par client\n\n"
                "Exemple : r=0.8, d=0.1, m=10£ ⇒ CLV = 10×0.8/(1+0.1−0.8) = 26.67£."
            )

    st.markdown("---")

    # Petite tendance CA dans le temps
    st.subheader("📈 Tendance de CA par mois (périmètre filtré)")
    monthly_rev = (
        df.groupby("InvoiceMonth")["Revenue"]
        .sum()
        .reset_index()
        .sort_values("InvoiceMonth")
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_rev["InvoiceMonth"], monthly_rev["Revenue"], marker="o")
    ax.set_title("CA mensuel")
    ax.set_xlabel("Mois")
    ax.set_ylabel("CA")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.caption(f"n mois = {len(monthly_rev)} | n transactions = {len(df):,}")
    
# ============================================================
# PAGE 2 : COHORTES
# ============================================================
elif page == "Cohortes (Diagnostiquer)":
    st.subheader("🧬 Cohortes d'acquisition & rétention")

    st.markdown(
        "Une *cohorte* regroupe les clients par date de première commande. "
        "On suit ensuite leur rétention et leur CA par *âge de cohorte* (M+0, M+1, ...)."
    )

    # Heatmap de rétention
    st.write("### 🔥 Heatmap de rétention par cohorte (en %)")
    retention_percent = retention_table.copy() * 100

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        retention_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        ax=ax1,
    )
    ax1.set_xlabel("Âge de cohorte (mois)")
    ax1.set_ylabel("Mois de cohorte")
    ax1.set_title("Rétention (%) par cohorte et âge (M+0 = mois d'acquisition)")
    st.pyplot(fig1)
    st.caption(f"n cohortes = {retention_table.shape[0]} | n âges = {retention_table.shape[1]}")

    # Focus sur une cohorte spécifique
    st.write("### 🎯 Focus sur une cohorte")
    cohort_list = list(retention_table.index.astype(str))
    selected_cohort = st.selectbox("Choisissez un mois de cohorte", cohort_list)

    if selected_cohort:
        cohort_idx = retention_table.index.astype(str) == selected_cohort
        retention_cohort = retention_table[cohort_idx].T.reset_index()
        retention_cohort.columns = ["CohortAge", "Retention"]

        revenue_cohort = revenue_age[cohort_idx].T.reset_index()
        revenue_cohort.columns = ["CohortAge", "Revenue"]

        col1, col2 = st.columns(2)

        with col1:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(retention_cohort["CohortAge"], retention_cohort["Retention"] * 100, marker="o")
            ax2.set_title(f"Rétention (%) - Cohorte {selected_cohort}")
            ax2.set_xlabel("Âge de cohorte (mois)")
            ax2.set_ylabel("Rétention (%)")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        with col2:
            fig3, ax3 = plt.subplots(figsize=(5, 4))
            ax3.bar(revenue_cohort["CohortAge"], revenue_cohort["Revenue"])
            ax3.set_title(f"CA par âge - Cohorte {selected_cohort}")
            ax3.set_xlabel("Âge de cohorte (mois)")
            ax3.set_ylabel("CA")
            st.pyplot(fig3)

        st.caption(
            "Une baisse forte de la rétention ou du CA après un certain âge de cohorte "
            "suggère un *décrochage* à cet âge (ex : M+2)."
        )
# ============================================================
# PAGE 3 : SEGMENTS RFM
# ============================================================
elif page == "Segments RFM (Prioriser)":
    st.subheader("👥 Segmentation RFM (Recency, Frequency, Monetary)")

    st.markdown(
        "RFM permet de prioriser les actions sur les clients :\n"
        "- *Recency* : nombre de jours depuis la dernière commande (plus petit = plus récent)\n"
        "- *Frequency* : nombre de factures différentes\n"
        "- *Monetary* : CA cumulé\n"
    )

    st.markdown("### 📋 Table RFM (échantillon)")
    st.dataframe(rfm.head(20))

    st.markdown("### 📊 Synthèse par segment RFM")
    rfm_summary = (
        rfm.groupby("Segment")
        .agg(
            n_customers=("Recency", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
        )
        .reset_index()
    )
    # jointure avec CA / marge réels (ici, approximations)
    rfm_summary["total_monetary"] = rfm_summary["n_customers"] * rfm_summary["avg_monetary"]

