import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    load_data,
    apply_filters,
    compute_cohorts,
    compute_cohort_metrics,
    compute_rfm,
    compute_rfm_trends,
    compute_clv_empirical,
    compute_clv_formula,
    compute_clv_probabilistic,
    simulate_scenarios,
    sensitivity_analysis,
    data_quality_report,
    compute_seasonality,
    compute_customer_segments_value,
    compute_churn_risk,
    compute_product_affinity,
    compute_cac_metrics,
    forecast_revenue,
    prepare_activation_list,
    generate_executive_summary
)

# ============================================================
# FONCTION HELPER POUR TOOLTIPS BLANCS
# ============================================================

def add_white_tooltip(fig):
    """
    Ajoute des tooltips blancs avec texte noir à un graphique Plotly.
    Utiliser cette fonction pour tous les graphiques qui ont un fond noir.
    """
    current_layout = fig.layout
    
    # Conserver les paramètres hoverlabel existants s'ils existent
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Inter",
            font_color="black",
            bordercolor="#667eea"
        )
    )
    return fig

# ============================================================
# CONFIGURATION PLOTLY - TEXTE EN NOIR
# ============================================================

# Template Plotly personnalisé pour texte noir
plotly_template = dict(
    layout=dict(
        font=dict(color='#0f172a', family='Inter, sans-serif', size=12),
        title=dict(font=dict(color='#0f172a', size=16, family='Inter, sans-serif')),
        xaxis=dict(
            title=dict(font=dict(color='#0f172a', size=14)),
            tickfont=dict(color='#0f172a', size=11)
        ),
        yaxis=dict(
            title=dict(font=dict(color='#0f172a', size=14)),
            tickfont=dict(color='#0f172a', size=11)
        ),
        legend=dict(font=dict(color='#0f172a', size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
)

# Appliquer le template par défaut
import plotly.io as pio
pio.templates["custom"] = plotly_template
pio.templates.default = "plotly_white+custom"

# ============================================================
# FONCTION UTILITAIRE POUR STYLE PLOTLY
# ============================================================

def apply_plotly_black_text(fig):
    """Applique un style avec texte noir à tous les graphiques Plotly"""
    fig.update_layout(
        font=dict(color='#0f172a', family='Inter, sans-serif', size=12),
        title_font=dict(color='#0f172a', size=16),
        xaxis=dict(
            title_font=dict(color='#0f172a', size=14),
            tickfont=dict(color='#0f172a', size=11),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title_font=dict(color='#0f172a', size=14),
            tickfont=dict(color='#0f172a', size=11),
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(font=dict(color='#0f172a', size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ============================================================
# FONCTION HELPER POUR TOOLTIPS
# ============================================================

def add_white_tooltip(fig):
    """Ajoute un style de tooltip blanc avec texte noir à tous les graphiques Plotly"""
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Inter",
            font_color="black",
            bordercolor="#667eea"
        )
    )
    return fig

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Marketing Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Dashboard d'analyse marketing - Cohortes, RFM & CLV"
    }
)

# ============================================================
# CSS PERSONNALISÉ - VERSION AMÉLIORÉE
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --accent: #ec4899;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
        --dark: #1e293b;
        --light: #f8fafc;
    }
    
    /* Layout principal */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(10px);
    }
    
    /* Texte - TOUT EN NOIR SUR FOND BLANC */
    .main p, .main span, .main div, .main label, .main li,
    .main strong, .main em, .main code, .stMarkdown, .stMarkdown * {
        color: #0f172a !important;
    }
    
    /* Titres */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.03em;
    }
    
    h2 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #0f172a;
        font-size: 2rem !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.2rem !important;
        border-bottom: 3px solid var(--primary);
        padding-bottom: 0.6rem;
    }
    
    h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1e293b;
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
    }
    
    /* Métriques */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="metric-container"] {
        background: white;
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.2);
        border-color: var(--primary);
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2.2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button *, .stButton > button span, .stButton > button p {
        color: white !important;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2.2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4);
    }
    
    .stDownloadButton > button *, .stDownloadButton > button span {
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"], [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] .row-widget label {
        color: white !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] select {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
        border-radius: 8px;
    }
    
    /* Alertes */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 1.2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    
    /* DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        border-radius: 10px;
        font-weight: 600;
        color: #0f172a !important;
        border: 1px solid #e2e8f0;
        padding: 0.8rem 1rem;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: var(--primary);
    }
    
    .streamlit-expanderHeader, .streamlit-expanderHeader *,
    .streamlit-expanderHeader p, .streamlit-expanderHeader span {
        color: #0f172a !important;
    }
    
    .streamlit-expanderContent, [data-testid="stExpander"] > div:last-child {
        background: white !important;
        padding: 1.2rem;
        border-radius: 0 0 10px 10px;
    }
    
    .streamlit-expanderContent, .streamlit-expanderContent p,
    .streamlit-expanderContent div, .streamlit-expanderContent span,
    .streamlit-expanderContent li {
        color: #0f172a !important;
    }
    
    [data-testid="stExpander"], [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *, details, details summary, details summary * {
        background: transparent !important;
        color: #0f172a !important;
    }
    
    [data-testid="stExpander"][open], details[open] {
        background: white !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Cards personnalisées */
    .custom-card {
        background: white;
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }
    
    .custom-card p, .custom-card span, .custom-card strong {
        color: #0f172a !important;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.3rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .badge, .badge * {
        color: white !important;
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .badge-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    
    .badge-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    
    /* Séparateur */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main > div {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #0f172a !important;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e2e8f0;
        border-color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: transparent !important;
    }
    
    /* Forcer le texte des tabs en noir sauf quand sélectionné */
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
        color: #0f172a !important;
    }
    
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) * {
        color: #0f172a !important;
    }
    
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) p {
        color: #0f172a !important;
    }
    
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) span {
        color: #0f172a !important;
    }
    
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) div {
        color: #0f172a !important;
    }
    
    /* Texte blanc pour le tab sélectionné */
    .stTabs [aria-selected="true"] * {
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] p {
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] span {
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] div {
        color: white !important;
    }
    
    /* Caption */
    .caption, small, .main small, [class*="caption"] {
        color: #64748b !important;
        font-size: 0.875rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    /* Plotly graphs - Forcer le texte en noir */
    .js-plotly-plot .plotly text {
        fill: #0f172a !important;
    }
    
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text,
    .js-plotly-plot .plotly .legendtext,
    .js-plotly-plot .plotly .g-gtitle text,
    .js-plotly-plot .plotly .g-xtitle text,
    .js-plotly-plot .plotly .g-ytitle text {
        fill: #0f172a !important;
        color: #0f172a !important;
    }
    
    /* Tabs - texte en noir */
    .stTabs [data-baseweb="tab"] {
        color: #0f172a !important;
    }
    
    .stTabs [data-baseweb="tab"] * {
        color: #0f172a !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    .stTabs [aria-selected="true"] * {
        color: white !important;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%);
        border-left: 4px solid var(--primary);
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box p, .info-box * {
        color: #1e293b !important;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.6rem !important; }
        h3 { font-size: 1.3rem !important; }
        .block-container { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITRE ET NAVIGATION
# ============================================================

st.title("📊 Marketing Analytics Dashboard Pro")

st.markdown("""
<div class="custom-card">
    <p style="margin: 0; font-size: 1.05rem;">
        <strong>🎯 Parcours recommandé :</strong> 
        <span class="badge badge-primary">📊 KPIs</span>
        <span class="badge badge-info">🧬 Cohortes</span>
        <span class="badge badge-success">👥 Segments RFM</span>
        <span class="badge badge-warning">🧪 Scénarios</span>
        <span class="badge badge-danger">📤 Export</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def load_raw_data():
    """Charge les données brutes avec cache"""
    df = load_data("data/raw/online_retail_cleaned.csv")
    return df

try:
    df_raw = load_raw_data()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des données : {e}")
    st.stop()

# ============================================================
# FILTRES GLOBAUX (SIDEBAR)
# ============================================================

st.sidebar.markdown("## 🔍 Filtres globaux")

# Dates
min_date = df_raw["InvoiceDate"].min().date()
max_date = df_raw["InvoiceDate"].max().date()

col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date = st.date_input("📅 Date de début", min_date, min_value=min_date, max_value=max_date)
with col_date2:
    end_date = st.date_input("📅 Date de fin", max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.sidebar.error("⚠️ La date de début doit être ≤ date de fin.")

# Pays
countries = st.sidebar.multiselect(
    "🌍 Pays",
    options=sorted(df_raw["Country"].unique()),
    default=["United Kingdom"] if "United Kingdom" in df_raw["Country"].unique() else [],
    help="Sélectionnez un ou plusieurs pays"
)

# Gestion des retours
returns_mode = st.sidebar.radio(
    "↩️ Gestion des retours",
    ["Inclure", "Exclure", "Neutraliser"],
    help=(
        "**Inclure** : les retours sont considérés comme des ventes négatives\n\n"
        "**Exclure** : les retours sont supprimés\n\n"
        "**Neutraliser** : le CA des retours est mis à 0"
    ),
)

# Seuil minimum de commande
min_order_value = st.sidebar.number_input(
    "💰 Seuil minimum de commande (£)",
    min_value=0.0,
    value=0.0,
    step=10.0,
    help="Filtre les transactions dont la valeur absolue est inférieure à ce seuil"
)

# Application des filtres
df = apply_filters(
    df_raw,
    start_date=start_date,
    end_date=end_date,
    countries=countries if len(countries) > 0 else None,
    returns_mode=returns_mode,
    min_order_value=min_order_value
)

# Vérification données filtrées
if df.empty:
    st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés. Veuillez ajuster vos filtres.")
    st.stop()

# Badge de synthèse des filtres
st.markdown(f"""
<div class="custom-card">
    <p style="margin: 0;">
        <strong>📅 Période :</strong> {start_date} → {end_date} &nbsp;|&nbsp; 
        <strong>📊 Transactions :</strong> {len(df):,} &nbsp;|&nbsp;
        <strong>👥 Clients :</strong> {df["Customer ID"].nunique():,} &nbsp;|&nbsp;
        <strong>💰 CA total :</strong> {df["Revenue"].sum():,.0f} £
    </p>
</div>
""", unsafe_allow_html=True)

if returns_mode == "Exclure":
    st.markdown('<span class="badge badge-warning">🟧 Retours exclus</span>', unsafe_allow_html=True)
elif returns_mode == "Neutraliser":
    st.markdown('<span class="badge badge-info">🟦 Retours neutralisés (CA = 0)</span>', unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# PRÉ-CALCULS COMMUNS
# ============================================================

# Cohortes
cohort_counts, retention_table, revenue_age, ltv_by_cohort = compute_cohorts(df)
avg_basket, purchase_freq = compute_cohort_metrics(df)

# RFM
rfm = compute_rfm(df)

# CLV
clv_emp_total = compute_clv_empirical(retention_table, revenue_age, horizon_months=12)
n_customers = df["Customer ID"].nunique()
clv_emp_per_cust = clv_emp_total / max(n_customers, 1)

# CLV formule
if retention_table.shape[1] > 1:
    r_hat = retention_table.iloc[:, 1].mean()
else:
    r_hat = retention_table.iloc[:, 0].mean()

horizon_months = (df["InvoiceMonth"].max().to_period("M") - df["InvoiceMonth"].min().to_period("M")).n + 1
total_revenue = df["Revenue"].sum()
avg_monthly_rev_per_cust = total_revenue / max(horizon_months * n_customers, 1)
margin_rate_default = 0.30
m_margin = avg_monthly_rev_per_cust * margin_rate_default
d_discount = 0.01
clv_formula_per_cust = compute_clv_formula(r=r_hat, d=d_discount, m=m_margin)

# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "📂 Navigation",
    [
        "📊 KPIs (Overview)",
        "🧬 Cohortes (Diagnostiquer)",
        "👥 Segments RFM (Prioriser)",
        "🧪 Scénarios (Simuler)",
        "📈 Analyses Avancées",
        "📤 Plan d'action & Export",
        "🧼 Qualité & Couverture"
    ],
)

# ============================================================
# PAGE 1 : KPIs (OVERVIEW)
# ============================================================

if page == "📊 KPIs (Overview)":
    st.markdown("## 📌 KPIs principaux")
    
    # Row 1 : KPIs principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    active_customers = n_customers
    total_transactions = df["Invoice"].nunique()
    
    with col1:
        st.metric("👥 Clients uniques", value=f"{active_customers:,}")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**Nombre de clients distincts** ayant passé au moins une commande sur la période filtrée.\n\n"
                "📘 **Exemple** : Si 3 clients A, B, C ont commandé → n = 3"
            )
    
    with col2:
        st.metric("🧾 Transactions", value=f"{total_transactions:,}")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**Nombre de factures uniques** (Invoice) sur la période.\n\n"
                "📘 **Exemple** : 1000 transactions sur le mois"
            )
    
    with col3:
        st.metric("💰 CA total", value=f"{total_revenue:,.0f} £")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**Chiffre d'affaires total** = Somme du Revenue sur la période.\n\n"
                "Revenue = Quantity × Price (les retours peuvent être négatifs ou neutralisés).\n\n"
                "📘 **Exemple** : Si 100 ventes de 50£ chacune → CA = 5,000£"
            )
    
    avg_basket_value = df.groupby("Invoice")["Revenue"].sum().mean()
    with col4:
        st.metric("🛒 Panier moyen", value=f"{avg_basket_value:,.2f} £")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**Montant moyen par transaction** = CA total / Nombre de transactions.\n\n"
                "📘 **Exemple** : 5,000£ de CA sur 100 transactions → Panier moyen = 50£"
            )
    
    avg_freq = df.groupby("Customer ID")["Invoice"].nunique().mean()
    with col5:
        st.metric("🔁 Fréquence moyenne", value=f"{avg_freq:.2f}")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**Nombre moyen de commandes par client** sur la période.\n\n"
                "📘 **Exemple** : 150 commandes pour 50 clients → Fréquence = 3"
            )
    
    st.markdown("---")
    
    # Row 2 : CLV metrics
    col6, col7, col8, col9 = st.columns(4)
    
    clv_3m = compute_clv_empirical(retention_table, revenue_age, horizon_months=min(3, revenue_age.shape[1]))
    clv_3m_per_cust = clv_3m / max(n_customers, 1)
    
    with col6:
        st.metric("📊 CA 90j / client", value=f"{clv_3m_per_cust:,.2f} £")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**CA moyen par client** sur les **3 premiers mois** (90 jours) après acquisition.\n\n"
                "Calculé en sommant le CA moyen par âge de cohorte sur M0, M+1, M+2.\n\n"
                "📘 **Exemple** : Mois 0 = 20£, Mois 1 = 10£, Mois 2 = 5£ → CLV_90j = 35£"
            )
    
    with col7:
        st.metric("📈 CLV empirique 12m", value=f"{clv_emp_per_cust:,.2f} £")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**CLV empirique** = somme du CA moyen par âge de cohorte sur 12 mois.\n\n"
                "Basée sur les données **historiques réelles** des cohortes passées.\n\n"
                "📘 **Exemple** : Si les clients dépensent en moyenne 10£/mois pendant 12 mois → CLV = 120£"
            )
    
    with col8:
        st.metric("🔮 CLV formule", value=f"{clv_formula_per_cust:,.2f} £")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**CLV via formule fermée** : CLV = m × r / (1 + d − r)\n\n"
                "- **r** : taux de rétention mensuel\n"
                "- **d** : taux d'actualisation mensuel\n"
                "- **m** : marge moyenne mensuelle\n\n"
                "📘 **Exemple** : r=0.8, d=0.1, m=10£ → CLV = 10 × 0.8 / (1.1 - 0.8) = 26.67£"
            )
    
    retention_m1 = retention_table.iloc[:, 1].mean() if retention_table.shape[1] > 1 else 0
    with col9:
        st.metric("🔄 Rétention M+1", value=f"{retention_m1*100:.1f}%")
        with st.expander("ℹ️ Définition"):
            st.write(
                "**Taux de rétention moyen à M+1** = % de clients qui font un achat 1 mois après leur première commande.\n\n"
                "📘 **Exemple** : Si 80 clients sur 100 reviennent après 1 mois → Rétention M+1 = 80%"
            )
    
    st.markdown("---")
    
    # Graphiques tendances
    st.markdown("## 📈 Évolution du CA")
    
    tab1, tab2, tab3 = st.tabs(["📊 CA mensuel", "📆 CA trimestriel", "🌍 CA par pays"])
    
    with tab1:
        monthly_rev = df.groupby("InvoiceMonth")["Revenue"].sum().reset_index()
        
        fig = px.line(
            monthly_rev,
            x="InvoiceMonth",
            y="Revenue",
            title="Évolution du CA mensuel",
            labels={"InvoiceMonth": "Mois", "Revenue": "CA (£)"}
        )
        fig.update_traces(
            line=dict(color="#667eea", width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.3)'
        )
        fig.update_layout(
            height=450,
            hovermode='x unified',
            plot_bgcolor='#0f172a',
            paper_bgcolor='#0f172a',
            font=dict(color='white', size=12),
            title=dict(
                text="Évolution du CA mensuel",
                font=dict(color='white', size=18, family='Inter')
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                color='white',
                title=dict(text="Mois", font=dict(color='white'))
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                color='white',
                title=dict(text="CA (£)", font=dict(color='white'))
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Inter",
                font_color="black",
                bordercolor="#667eea"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"📊 {len(monthly_rev)} mois analysés | {len(df):,} transactions")
    
    with tab2:
        quarterly_rev = df.groupby("Quarter")["Revenue"].sum().reset_index()
        
        fig = px.bar(
            quarterly_rev,
            x="Quarter",
            y="Revenue",
            title="CA trimestriel",
            labels={"Quarter": "Trimestre", "Revenue": "CA (£)"},
            color="Revenue",
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            height=450,
            plot_bgcolor='#0f172a',
            paper_bgcolor='#0f172a',
            font=dict(color='white', size=12),
            title=dict(
                text="CA trimestriel",
                font=dict(color='white', size=18, family='Inter')
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                color='white',
                title=dict(text="Trimestre", font=dict(color='white'))
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                color='white',
                title=dict(text="CA (£)", font=dict(color='white'))
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Inter",
                font_color="black",
                bordercolor="#667eea"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        country_rev = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False).head(10).reset_index()
        
        fig = px.bar(
            country_rev,
            x="Revenue",
            y="Country",
            orientation='h',
            title="Top 10 pays par CA",
            labels={"Country": "Pays", "Revenue": "CA (£)"},
            color="Revenue",
            color_continuous_scale="plasma"
        )
        fig.update_layout(
            height=450,
            plot_bgcolor='#0f172a',
            paper_bgcolor='#0f172a',
            font=dict(color='white', size=12),
            title=dict(
                text="Top 10 pays par CA",
                font=dict(color='white', size=18, family='Inter')
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                color='white',
                title=dict(text="CA (£)", font=dict(color='white'))
            ),
            yaxis=dict(
                showgrid=False,
                color='white',
                title=dict(text="Pays", font=dict(color='white'))
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Inter",
                font_color="black",
                bordercolor="#667eea"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 : COHORTES
# ============================================================

elif page == "🧬 Cohortes (Diagnostiquer)":
    st.markdown("## 🧬 Analyse des cohortes d'acquisition")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0;">
            <strong>🎯 Qu'est-ce qu'une cohorte ?</strong><br>
            Une cohorte regroupe les clients par <strong>mois de première commande</strong>. 
            On suit ensuite leur rétention et leur CA par <strong>âge de cohorte</strong> (M+0, M+1, M+2, ...).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Heatmap de rétention
    st.markdown("### 🔥 Heatmap de rétention par cohorte")
    
    retention_percent = retention_table.copy() * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=retention_percent.values,
        x=[f"M+{i}" for i in range(retention_percent.shape[1])],
        y=retention_percent.index.strftime("%Y-%m"),
        colorscale='RdYlGn',
        text=retention_percent.values,
        texttemplate='%{text:.1f}%',
        textfont={"size": 10, "color": "black"},
        colorbar=dict(title="Rétention (%)")
    ))
    
    fig.update_layout(
        title="Rétention (%) par cohorte et âge - M+0 = mois d'acquisition",
        xaxis_title="Âge de cohorte",
        yaxis_title="Mois de cohorte",
        height=600,
        font=dict(size=12)
    )
    
    fig = add_white_tooltip(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"📊 {retention_table.shape[0]} cohortes | {retention_table.shape[1]} âges analysés")
    
    with st.expander("ℹ️ Comment interpréter cette heatmap ?"):
        st.write("""
        **Lecture de la heatmap :**
        
        - **Axe vertical (Y)** : Mois de cohorte (date de première commande)
        - **Axe horizontal (X)** : Âge de la cohorte en mois (M+0, M+1, M+2, ...)
        - **Couleur** : Taux de rétention (vert = élevé, rouge = faible)
        
        **🔍 Insights à rechercher :**
        - Lignes qui deviennent rapidement rouges → cohortes qui décrochent vite
        - Colonnes rouges → âges critiques où tous les clients décrochent
        - Diagonale verte → bonnes cohortes qui restent fidèles
        
        **📘 Exemple :** Si la cohorte 2020-01 est à 80% en M+0, 50% en M+1, 30% en M+2,
        cela signifie que 50% des clients ont repassé commande après 1 mois.
        """)
    
    st.markdown("---")
    
    # Focus sur une cohorte
    st.markdown("### 🎯 Focus détaillé sur une cohorte")
    
    cohort_list = list(retention_table.index.strftime("%Y-%m"))
    selected_cohort_str = st.selectbox("📅 Choisissez un mois de cohorte", cohort_list)
    
    if selected_cohort_str:
        selected_cohort = pd.to_datetime(selected_cohort_str)
        cohort_idx = retention_table.index == selected_cohort
        
        retention_cohort = retention_table[cohort_idx].T.reset_index()
        retention_cohort.columns = ["CohortAge", "Retention"]
        
        revenue_cohort = revenue_age[cohort_idx].T.reset_index()
        revenue_cohort.columns = ["CohortAge", "Revenue"]
        
        ltv_cohort = ltv_by_cohort[cohort_idx].T.reset_index()
        ltv_cohort.columns = ["CohortAge", "LTV_Cumul"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=retention_cohort["CohortAge"],
                y=retention_cohort["Retention"] * 100,
                mode='lines+markers',
                name='Rétention',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            fig.update_layout(
                title=f"📉 Rétention - Cohorte {selected_cohort_str}",
                xaxis_title="Âge de cohorte (mois)",
                yaxis_title="Rétention (%)",
                height=400,
                hovermode='x unified'
            )
            fig = add_white_tooltip(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=revenue_cohort["CohortAge"],
                y=revenue_cohort["Revenue"],
                name='CA',
                marker=dict(color='#764ba2'),
                text=revenue_cohort["Revenue"].round(0),
                textposition='outside'
            ))
            fig.update_layout(
                title=f"💰 CA par âge - Cohorte {selected_cohort_str}",
                xaxis_title="Âge de cohorte (mois)",
                yaxis_title="CA (£)",
                height=400,
                hovermode='x unified'
            )
            fig = add_white_tooltip(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # LTV cumulée
        st.markdown("#### 📊 LTV cumulée par âge")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ltv_cohort["CohortAge"],
            y=ltv_cohort["LTV_Cumul"],
            mode='lines+markers',
            name='LTV cumulée',
            line=dict(color='#10b981', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.2)'
        ))
        fig.update_layout(
            title=f"LTV cumulée - Cohorte {selected_cohort_str}",
            xaxis_title="Âge de cohorte (mois)",
            yaxis_title="LTV cumulée (£)",
            height=350
        )
        fig = add_white_tooltip(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <p style="color: #64748b; font-size: 0.875rem; font-style: italic;">
        💡 Une <strong>baisse forte de la rétention</strong> ou du CA après un certain âge 
        suggère un <strong>décrochage à cet âge</strong> (ex : M+2). 
        La LTV cumulée montre la <strong>valeur totale apportée</strong> par la cohorte au fil du temps.
        </p>
        """, unsafe_allow_html=True)

# ============================================================
# PAGE 3 : SEGMENTS RFM
# ============================================================

elif page == "👥 Segments RFM (Prioriser)":
    st.markdown("## 👥 Segmentation RFM")
    st.markdown("### (Recency, Frequency, Monetary)")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0;">
            <strong>🎯 Qu'est-ce que le RFM ?</strong><br>
            La segmentation <strong>RFM</strong> permet de prioriser les actions marketing :<br><br>
            • <strong>Recency (R)</strong> : Nombre de jours depuis la dernière commande (plus petit = plus récent)<br>
            • <strong>Frequency (F)</strong> : Nombre de factures différentes<br>
            • <strong>Monetary (M)</strong> : CA cumulé du client
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Distribution des segments
    st.markdown("### 📊 Distribution des segments RFM")
    
    segment_counts = rfm["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Count"]
    
    fig = px.pie(
        segment_counts,
        names="Segment",
        values="Count",
        title="Répartition des clients par segment RFM",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=500)
    fig = add_white_tooltip(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"📊 {len(rfm)} clients segmentés | {segment_counts.shape[0]} segments identifiés")
    
    st.markdown("---")
    
    # Métriques par segment
    st.markdown("### 💰 Valeur par segment RFM")
    
    segment_metrics = compute_customer_segments_value(df, rfm)
    segment_metrics = segment_metrics.sort_values("CA_Total", ascending=False)
    
    # Affichage table
    st.dataframe(
        segment_metrics.style.format({
            "NbClients": "{:,.0f}",
            "NbCommandes": "{:,.0f}",
            "CA_Total": "{:,.2f} £",
            "PanierMoyen": "{:,.2f} £",
            "QteTotale": "{:,.0f}",
            "CA_Par_Client": "{:,.2f} £"
        }).background_gradient(subset=["CA_Total"], cmap="Greens"),
        use_container_width=True
    )
    
    with st.expander("ℹ️ Définitions des colonnes"):
        st.write("""
        **Colonnes du tableau :**
        
        - **NbClients** : Nombre de clients dans le segment
        - **NbCommandes** : Nombre total de commandes du segment
        - **CA_Total** : Chiffre d'affaires total généré par le segment
        - **PanierMoyen** : Montant moyen par transaction
        - **QteTotale** : Quantité totale de produits achetés
        - **CA_Par_Client** : CA moyen par client du segment
        
        **🎯 Comment utiliser ces données :**
        - Priorisez les segments à **CA_Total élevé** (Champions, Loyaux)
        - Identifiez les segments à **réactiver** (À risque, Hibernants)
        - Segmentez vos campagnes selon la **valeur client**
        """)
    
    st.markdown("---")
    
    # Graphiques comparatifs
    st.markdown("### 📈 Comparaisons visuelles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            segment_metrics,
            x="Segment",
            y="CA_Total",
            title="CA total par segment RFM",
            labels={"Segment": "Segment", "CA_Total": "CA total (£)"},
            color="CA_Total",
            color_continuous_scale="Viridis",
            text="CA_Total"
        )
        fig.update_traces(texttemplate='%{text:,.0f}£', textposition='outside')
        fig.update_layout(height=450)
        fig = add_white_tooltip(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            segment_metrics,
            x="NbClients",
            y="CA_Par_Client",
            size="CA_Total",
            color="Segment",
            title="Nombre de clients vs CA par client",
            labels={
                "NbClients": "Nombre de clients",
                "CA_Par_Client": "CA par client (£)",
                "Segment": "Segment"
            },
            hover_data=["CA_Total"]
        )
        fig.update_layout(height=450)
        fig = add_white_tooltip(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <p style="color: #64748b; font-size: 0.875rem; font-style: italic;">
    💡 <strong>Insights :</strong> Les segments "Champions" et "Loyaux" génèrent souvent 70-80% du CA. 
    Investissez sur ces segments tout en réactivant les segments "À risque" avant qu'ils ne deviennent "Perdus".
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risque de churn
    st.markdown("### ⚠️ Analyse du risque de churn")
    
    rfm_churn = compute_churn_risk(rfm, threshold_days=90)
    
    churn_dist = rfm_churn["ChurnRisk"].value_counts().reset_index()
    churn_dist.columns = ["Risque", "Count"]
    
    col1, col2, col3 = st.columns(3)
    
    high_risk = (rfm_churn["ChurnRisk"] == "Élevé").sum()
    medium_risk = (rfm_churn["ChurnRisk"] == "Moyen").sum()
    low_risk = (rfm_churn["ChurnRisk"] == "Faible").sum()
    
    col1.metric("🔴 Risque élevé", f"{high_risk:,}", help="Clients n'ayant pas commandé depuis 90+ jours")
    col2.metric("🟡 Risque moyen", f"{medium_risk:,}", help="Clients n'ayant pas commandé depuis 45-90 jours")
    col3.metric("🟢 Risque faible", f"{low_risk:,}", help="Clients actifs (< 45 jours)")
    
    fig = px.bar(
        churn_dist,
        x="Risque",
        y="Count",
        title="Distribution du risque de churn",
        color="Risque",
        color_discrete_map={"Élevé": "#ef4444", "Moyen": "#f59e0b", "Faible": "#10b981"}
    )
    fig.update_layout(height=400)
    fig = add_white_tooltip(fig)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 4 : SCÉNARIOS
# ============================================================

elif page == "🧪 Scénarios (Simuler)":
    st.markdown("## 🧪 Simulation de scénarios")
    st.markdown("### Impact sur CLV, Rétention & Marge")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0;">
            <strong>🎯 Objectif :</strong> Simuler l'impact de changements stratégiques sur la CLV.<br><br>
            On part d'une <strong>situation baseline</strong> (rétention, marge, actualisation) 
            et on teste l'effet de :<br>
            • Un <strong>gain de rétention</strong> (+r%)<br>
            • Un <strong>changement de marge</strong> (± marge%)<br>
            • Une <strong>remise moyenne</strong> (impact sur marge)<br><br>
            <strong>Résultat :</strong> Calcul de ΔCLV, ΔCA, Δrétention
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Paramètres baseline
    st.markdown("### ⚙️ Paramètres baseline (situation actuelle)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_r = st.slider(
            "📊 Taux de rétention mensuel (r)",
            min_value=0.0,
            max_value=0.99,
            value=float(np.round(r_hat, 2)),
            step=0.01,
            help="Proportion de clients qui restent actifs chaque mois"
        )
    
    with col2:
        base_d = st.slider(
            "💹 Taux d'actualisation mensuel (d)",
            min_value=0.0,
            max_value=0.2,
            value=d_discount,
            step=0.01,
            help="Taux d'actualisation financier (valeur temps de l'argent)"
        )
    
    with col3:
        base_margin = st.number_input(
            "💰 Marge mensuelle par client (£)",
            min_value=0.0,
            value=float(np.round(m_margin, 2)),
            step=1.0,
            help="Marge nette moyenne par client et par mois"
        )
    
    base_clv_formula = compute_clv_formula(base_r, base_d, base_margin)
    
    st.markdown(f"""
    <div class="custom-card">
        <p style="margin: 0;">
            <strong>📊 CLV Baseline :</strong> <span style="font-size: 1.5rem; font-weight: bold; color: #667eea;">{base_clv_formula:,.2f} £</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Paramètres du scénario
    st.markdown("### 🔄 Paramètres du scénario à simuler")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        retention_gain_pct = st.slider(
            "📈 Gain de rétention (%)",
            min_value=-20,
            max_value=50,
            value=5,
            step=1,
            help="Variation du taux de rétention. Ex : +5% améliore r de 5 points de pourcentage"
        )
    
    with col2:
        margin_gain_pct = st.slider(
            "💵 Variation de la marge (%)",
            min_value=-50,
            max_value=50,
            value=0,
            step=1,
            help="Variation de la marge nette. Ex : +10% améliore la marge de 10%"
        )
    
    with col3:
        discount_pct = st.slider(
            "🏷️ Remise moyenne (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="Remise commerciale moyenne appliquée, réduit la marge"
        )
    
    with col4:
        scenario_segment = st.selectbox(
            "🎯 Segment RFM cible",
            options=["Tous"] + sorted(rfm["Segment"].unique().tolist()),
            help="Appliquer le scénario uniquement à un segment spécifique"
        )
    
    # Calcul du scénario
    total_margin_change = margin_gain_pct - discount_pct
    retention_gain = retention_gain_pct / 100
    margin_gain = total_margin_change / 100
    
    new_clv_formula, delta_clv, metrics = simulate_scenarios(
        base_clv=base_clv_formula,
        retention=base_r,
        margin=base_margin,
        retention_gain=retention_gain,
        margin_gain=margin_gain,
        discount=base_d
    )
    
    st.markdown("---")
    
    # Résultats du scénario
    st.markdown("### 📈 Résultats de la simulation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📊 CLV Baseline",
            f"{metrics['base_clv']:,.2f} £",
            help="CLV avec les paramètres actuels"
        )
    
    with col2:
        delta_color = "normal" if metrics['delta_clv'] >= 0 else "inverse"
        st.metric(
            "🎯 CLV Scénario",
            f"{metrics['new_clv']:,.2f} £",
            delta=f"{metrics['delta_clv']:,.2f} £ ({metrics['delta_pct']:.1f}%)",
            delta_color=delta_color,
            help="CLV après application du scénario"
        )
    
    with col3:
        st.metric(
            "🔄 Nouvelle rétention",
            f"{metrics['new_retention']*100:.1f}%",
            delta=f"{(metrics['new_retention'] - base_r)*100:.2f} pts",
            help="Taux de rétention après scénario"
        )
    
    with col4:
        st.metric(
            "💰 Nouvelle marge",
            f"{metrics['new_margin']:,.2f} £",
            delta=f"{(metrics['new_margin'] - base_margin):.2f} £",
            help="Marge mensuelle après scénario"
        )
    
    # Impact détaillé
    st.markdown("#### 🔍 Décomposition de l'impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <p><strong>📈 Impact rétention :</strong> {:.2f} £</p>
            <p><strong>💰 Impact marge :</strong> {:.2f} £</p>
            <p><strong>🎯 Impact total :</strong> {:.2f} £</p>
        </div>
        """.format(
            metrics['retention_impact'],
            metrics['margin_impact'],
            metrics['delta_clv']
        ), unsafe_allow_html=True)
    
    with col2:
        impact_data = pd.DataFrame({
            "Levier": ["Rétention", "Marge"],
            "Impact (£)": [metrics['retention_impact'], metrics['margin_impact']]
        })
        
        fig = px.bar(
            impact_data,
            x="Levier",
            y="Impact (£)",
            title="Contribution de chaque levier",
            color="Impact (£)",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(height=300)
        fig = add_white_tooltip(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Extrapolation sur la base clients
    if scenario_segment == "Tous":
        total_impact = metrics['delta_clv'] * n_customers
        st.markdown(f"""
        <div class="custom-card" style="background: linear-gradient(135deg, #e0e7ff 0%, #ddd6fe 100%);">
            <p style="margin: 0; font-size: 1.2rem;">
                <strong>💡 Impact extrapolé sur toute la base :</strong><br>
                <span style="font-size: 1.8rem; font-weight: bold; color: #667eea;">
                {total_impact:,.0f} £
                </span><br>
                <small>({n_customers:,} clients × {metrics['delta_clv']:.2f} £/client)</small>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analyse de sensibilité
    st.markdown("### 🎚️ Analyse de sensibilité")
    
    with st.expander("📊 Afficher l'analyse de sensibilité complète"):
        st.markdown("""
        Cette analyse montre comment la CLV varie selon différentes combinaisons de 
        **taux de rétention** et de **marge**.
        """)
        
        sensitivity_df = sensitivity_analysis(
            retention=base_r,
            margin=base_margin,
            discount=base_d,
            retention_range=(-20, 30),
            margin_range=(-30, 30),
            steps=15
        )
        
        # Heatmap de sensibilité
        pivot_sens = sensitivity_df.pivot_table(
            index="RetentionPct",
            columns="MarginPct",
            values="CLV"
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_sens.values,
            x=pivot_sens.columns,
            y=pivot_sens.index,
            colorscale='RdYlGn',
            colorbar=dict(title="CLV (£)")
        ))
        
        fig.update_layout(
            title="Sensibilité de la CLV (Rétention × Marge)",
            xaxis_title="Variation de marge (%)",
            yaxis_title="Variation de rétention (%)",
            height=600
        )
        
        fig = add_white_tooltip(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("🔍 Les zones vertes indiquent des combinaisons favorables (CLV élevée)")

# ============================================================
# PAGE 5 : ANALYSES AVANCÉES
# ============================================================

elif page == "📈 Analyses Avancées":
    st.markdown("## 📈 Analyses avancées & Insights")
    
    tabs = st.tabs([
        "🌡️ Saisonnalité",
        "🛍️ Affinité produits",
        "💼 CAC & LTV/CAC"
    ])
    
    # TAB 1 : Saisonnalité
    with tabs[0]:
        st.markdown("### 🌡️ Analyse de saisonnalité")
        
        monthly_rev, daily_rev, hourly_rev = compute_seasonality(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=monthly_rev.index,
                y=monthly_rev.values,
                labels={"x": "Mois", "y": "CA (£)"},
                title="CA par mois de l'année",
                color=monthly_rev.values,
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=400)
            fig = add_white_tooltip(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=hourly_rev.index,
                y=hourly_rev.values,
                labels={"x": "Heure", "y": "CA (£)"},
                title="CA par heure de la journée",
                color=hourly_rev.values,
                color_continuous_scale="Plasma"
            )
            fig.update_layout(height=400)
            fig = add_white_tooltip(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <p style="color: #64748b; font-size: 0.875rem; font-style: italic;">
        💡 Identifiez les <strong>pics saisonniers</strong> pour optimiser vos campagnes marketing 
        et votre gestion des stocks.
        </p>
        """, unsafe_allow_html=True)
    
    # TAB 2 : Affinité produits
    with tabs[1]:
        st.markdown("### 🛍️ Affinité produits")
        
        if "StockCode" in df.columns:
            affinity = compute_product_affinity(df)
            
            if not affinity.empty:
                st.markdown("#### Top 20 paires de produits achetés ensemble")
                
                top_20 = affinity.head(20)
                
                fig = px.bar(
                    top_20,
                    x="Frequency",
                    y=top_20["Product1"] + " + " + top_20["Product2"],
                    orientation='h',
                    title="Produits fréquemment achetés ensemble",
                    labels={"y": "Paire de produits", "Frequency": "Fréquence"},
                    color="Frequency",
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(height=600)
                fig = add_white_tooltip(fig)
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(affinity.head(50), use_container_width=True)
                
                st.markdown("""
                <p style="color: #64748b; font-size: 0.875rem; font-style: italic;">
                💡 Utilisez ces insights pour créer des <strong>bundles</strong>, 
                des <strong>recommandations produits</strong> ou des <strong>promotions croisées</strong>.
                </p>
                """, unsafe_allow_html=True)
            else:
                st.info("Pas assez de données pour calculer les affinités produits.")
        else:
            st.warning("La colonne 'StockCode' n'est pas disponible dans les données.")
    
    # TAB 3 : CAC & LTV/CAC
    with tabs[2]:
        st.markdown("### 💼 CAC (Customer Acquisition Cost) & Ratio LTV/CAC")
        
        st.markdown("""
        <div class="info-box">
            <p style="margin: 0;">
                Le <strong>CAC</strong> (Customer Acquisition Cost) représente le coût d'acquisition d'un nouveau client.<br>
                Le ratio <strong>LTV/CAC</strong> mesure la rentabilité de vos investissements marketing.<br><br>
                <strong>🎯 Benchmark :</strong> Un ratio LTV/CAC > 3 est généralement considéré comme excellent.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        marketing_budget = st.number_input(
            "💰 Budget marketing total (£) - optionnel",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Entrez votre budget marketing pour calculer le CAC et le ratio LTV/CAC"
        )
        
        if marketing_budget > 0:
            cac_metrics = compute_cac_metrics(df, marketing_spend=marketing_budget)
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("👥 Nouveaux clients", f"{cac_metrics['total_new_customers']:,}")
            col2.metric("💰 CAC", f"{cac_metrics['cac']:,.2f} £")
            col3.metric("📊 Ratio LTV/CAC", f"{cac_metrics['ltv_cac_ratio']:.2f}x")
            col4.metric("⏱️ Break-even", f"{cac_metrics['break_even_months']:.1f} mois")
            
            # Visualisation du ratio
            ratio = cac_metrics['ltv_cac_ratio']
            
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=ratio,
                delta={'reference': 3, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 1], 'color': "#fee2e2"},
                        {'range': [1, 3], 'color': "#fef3c7"},
                        {'range': [3, 10], 'color': "#d1fae5"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 3
                    }
                },
                title={'text': "Ratio LTV/CAC"}
            ))
            
            fig.update_layout(height=400)
            fig = add_white_tooltip(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            if ratio >= 3:
                st.success("✅ Excellent ! Votre ratio LTV/CAC est supérieur à 3. Vos investissements marketing sont rentables.")
            elif ratio >= 1:
                st.warning("⚠️ Attention ! Votre ratio LTV/CAC est entre 1 et 3. Vous devriez optimiser vos coûts d'acquisition ou augmenter la LTV.")
            else:
                st.error("❌ Critique ! Votre ratio LTV/CAC est inférieur à 1. Vous perdez de l'argent sur chaque acquisition client.")
        
        else:
            st.info("💡 Entrez un budget marketing pour calculer le CAC et le ratio LTV/CAC.")

# ============================================================
# PAGE 6 : EXPORT
# ============================================================

elif page == "📤 Plan d'action & Export":
    st.markdown("## 📤 Plan d'action & Exports")
    
    st.markdown("""
    <div class="info-box">
        <p style="margin: 0;">
            <strong>🎯 Objectif :</strong> Passer du diagnostic à l'exécution.<br><br>
            • Exporter une <strong>liste activable</strong> (CustomerID, segment RFM, métriques clés)<br>
            • Exporter les <strong>données filtrées</strong><br>
            • Exporter des <strong>graphiques en PNG</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section 1 : Liste activable
    st.markdown("### 📋 Liste activable pour campagnes CRM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segments_to_activate = st.multiselect(
            "🎯 Segments RFM à activer",
            options=sorted(rfm["Segment"].unique()),
            default=["Champions", "Loyaux"],
            help="Sélectionnez les segments à cibler dans vos campagnes"
        )
    
    with col2:
        top_n = st.number_input(
            "📊 Limiter à N clients (0 = tous)",
            min_value=0,
            value=0,
            step=100,
            help="Limitez l'export aux N meilleurs clients (par valeur)"
        )
    
    if segments_to_activate:
        activable = prepare_activation_list(
            rfm,
            segments_to_activate=segments_to_activate,
            top_n=top_n if top_n > 0 else None
        )
        
        st.markdown(f"**📊 {len(activable)} clients sélectionnés**")
        st.dataframe(activable.head(20), use_container_width=True)
        
        # Export CSV
        csv_activable = activable.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Télécharger la liste activable (CSV)",
            data=csv_activable,
            file_name=f"liste_activable_rfm_{len(activable)}_clients.csv",
            mime="text/csv",
        )
        
        # Statistiques de la liste
        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Clients", f"{len(activable):,}")
        col2.metric("💰 CA potentiel", f"{activable['Monetary'].sum():,.0f} £")
        col3.metric("📊 CA moyen/client", f"{activable['Monetary'].mean():,.2f} £")
    
    else:
        st.info("👆 Sélectionnez au moins un segment RFM pour générer la liste activable.")
    
    st.markdown("---")
    
    # Section 2 : Export données filtrées
    st.markdown("### 🗂️ Export des données filtrées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_filtered = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Télécharger les transactions filtrées (CSV)",
            data=csv_filtered,
            file_name="transactions_filtrees.csv",
            mime="text/csv",
        )
    
    with col2:
        csv_rfm = rfm.to_csv().encode("utf-8")
        st.download_button(
            label="💾 Télécharger la table RFM complète (CSV)",
            data=csv_rfm,
            file_name="rfm_complete.csv",
            mime="text/csv",
        )
    
    st.markdown("---")
    
    # Section 3 : Export graphiques
    st.markdown("### 🖼️ Export de visualisations")
    
    st.markdown("#### Heatmap de rétention")
    
    retention_percent = retention_table.copy() * 100
    
    fig_export, ax_export = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        retention_percent,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        ax=ax_export,
        linewidths=0.5,
        linecolor='white',
        annot_kws={"size": 10, "weight": "bold", "color": "black"},
        cbar_kws={'label': 'Rétention (%)'}
    )
    ax_export.set_xlabel("Âge de cohorte (mois)", fontsize=14, fontweight='bold')
    ax_export.set_ylabel("Mois de cohorte", fontsize=14, fontweight='bold')
    ax_export.set_title(
        f"Rétention par cohorte - Période {start_date} → {end_date}",
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    plt.tight_layout()
    st.pyplot(fig_export)
    
    buf = io.BytesIO()
    fig_export.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    buf.seek(0)
    
    st.download_button(
        label="📥 Télécharger la heatmap (PNG haute résolution)",
        data=buf,
        file_name=f"heatmap_retention_{start_date}_{end_date}.png",
        mime="image/png",
    )
    
    st.markdown("---")
    
    # Section 4 : Résumé exécutif
    st.markdown("### 📊 Résumé exécutif")
    
    summary = generate_executive_summary(df, rfm, cohort_counts, retention_table)
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("👥 Clients totaux", f"{summary['total_customers']:,}")
    col2.metric("🧾 Transactions", f"{summary['total_transactions']:,}")
    col3.metric("💰 CA total", f"{summary['total_revenue']:,.0f} £")
    
    col1.metric("🛒 Panier moyen", f"{summary['avg_basket']:,.2f} £")
    col2.metric("🔁 Fréquence moyenne", f"{summary['avg_purchase_frequency']:.2f}")
    col3.metric("🔄 Rétention M+1", f"{summary['avg_retention_m1']*100:.1f}%")
    
    st.markdown("#### 📈 Distribution RFM")
    
    rfm_dist_df = pd.DataFrame(list(summary['rfm_distribution'].items()), columns=["Segment", "Count"])
    
    fig = px.pie(
        rfm_dist_df,
        names="Segment",
        values="Count",
        title="Répartition des clients par segment",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(height=400)
    fig = add_white_tooltip(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""
    <div class="custom-card">
        <p style="margin: 0;">
            <strong>🏆 Segment #1 par CA :</strong> <span class="badge badge-success">{summary['top_segment']}</span><br>
            <strong>📅 Période analysée :</strong> {summary['date_start'].strftime('%Y-%m-%d')} → {summary['date_end'].strftime('%Y-%m-%d')} 
            ({summary['analysis_days']} jours)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Export résumé JSON
    import json
    summary_json = json.dumps(summary, default=str, indent=2)
    st.download_button(
        label="📥 Télécharger le résumé exécutif (JSON)",
        data=summary_json,
        file_name="resume_executif.json",
        mime="application/json"
    )

# ============================================================
# PAGE 7 : QUALITÉ
# ============================================================

elif page == "🧼 Qualité & Couverture":
    st.markdown("## 🧼 Qualité des données & Couverture")
    
    dq = data_quality_report(df_raw, df)
    
    st.markdown("### 📊 Métriques de volumétrie")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("📦 Lignes totales", f"{dq['rows_total']:,}")
    col2.metric("✅ Lignes après filtres", f"{dq['rows_filtered']:,}")
    col3.metric("📊 % gardées", f"{dq['pct_kept']:.1f}%")
    col4.metric("👥 Clients", f"{dq['customers_filtered']:,}")
    
    st.markdown("---")
    
    st.markdown("### 🔍 Qualité des données")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("❓ % CustomerID manquants", f"{dq['pct_missing_customer']:.2f}%")
    col2.metric("↩️ % Factures de retours", f"{dq['pct_returns']:.2f}%")
    col3.metric("💰 % CA lié aux retours", f"{abs(dq['returns_share_revenue']):.2f}%")
    
    col1.metric("⚠️ % Outliers (IQR)", f"{dq['pct_outliers']:.2f}%")
    col2.metric("💷 CA total brut", f"{dq['total_revenue']:,.0f} £")
    col3.metric("💸 CA retours", f"{abs(dq['returns_revenue']):,.0f} £")
    
    st.markdown("---")
    
    st.markdown("### 📅 Couverture temporelle")
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("📅 Date min", dq['date_min'].strftime('%Y-%m-%d'))
    col2.metric("📅 Date max", dq['date_max'].strftime('%Y-%m-%d'))
    col3.metric("📊 Jours couverts", f"{dq['days_coverage']:,}")
    
    st.markdown("---")
    
    # Visualisation de la qualité
    st.markdown("### 📈 Visualisation de la qualité")
    
    quality_metrics = {
        "Complétude": 100 - dq['pct_missing_customer'],
        "Fiabilité": 100 - dq['pct_outliers'],
        "Couverture": dq['pct_kept']
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(quality_metrics.keys()),
        y=list(quality_metrics.values()),
        marker=dict(color=['#10b981', '#3b82f6', '#f59e0b']),
        text=[f"{v:.1f}%" for v in quality_metrics.values()],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Score de qualité des données (%)",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 110]),
        height=400,
        showlegend=False
    )
    
    fig = add_white_tooltip(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <p style="color: #64748b; font-size: 0.875rem; font-style: italic;">
    💡 Cette page permet de juger la <strong>fiabilité</strong> des analyses : 
    volume de données, impact des filtres, importance des retours, présence d'outliers, etc.
    </p>
    """, unsafe_allow_html=True)
    
    # Recommandations
    with st.expander("💡 Recommandations d'amélioration"):
        st.write("**Suggestions pour améliorer la qualité des données :**")
        
        recommendations = []
        
        if dq['pct_missing_customer'] > 5:
            recommendations.append("⚠️ Plus de 5% de CustomerID manquants. Vérifiez la source de données.")
        
        if dq['pct_outliers'] > 10:
            recommendations.append("⚠️ Plus de 10% d'outliers détectés. Envisagez un nettoyage supplémentaire.")
        
        if abs(dq['returns_share_revenue']) > 20:
            recommendations.append("⚠️ Les retours représentent plus de 20% du CA. Analysez les causes.")
        
        if dq['pct_kept'] < 80:
            recommendations.append("⚠️ Moins de 80% des données conservées après filtres. Vérifiez vos critères de filtrage.")
        
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success("✅ La qualité des données est bonne ! Aucune recommandation particulière.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>📊 Marketing Analytics Dashboard Pro</strong><br>
        Cohortes • RFM • CLV • Scénarios • Analyses Avancées
    </p>
    <p style="margin-top: 0.5rem; font-size: 0.8rem;">
        Powered by Streamlit • Data from Online Retail II (UCI)
    </p>
</div>
""", unsafe_allow_html=True)
