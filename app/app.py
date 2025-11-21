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
# ==========================================================
# CONFIG STREAMLIT
# ==========================================================

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

start_date =st.sidebar.date_input("Date de début", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("Date de fin", max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("La date de début doit être ≤ date de fin.")

countries = st.sidebar.multiselect(
    "Pays",
    options=sorted(df_raw["Country"].unique()),
    default=["United Kingdom"] if "United Kingdom" in df_raw["Country"].unique() else [],
)

returns_mode =st.sidebar.radio(
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







