# 📊 Marketing Analytics Dashboard Pro

> Dashboard interactif d'analyse marketing basé sur les cohortes, la segmentation RFM et le CLV

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.17+-green.svg)

---

## 🎯 Description

Dashboard de DataViz développé pour **analyser et optimiser les stratégies marketing** d'un e-commerce UK. Permet de diagnostiquer la rétention, segmenter les clients, calculer la valeur client (CLV) et simuler des scénarios business.

**Dataset** : [Online Retail II (UCI)](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) - 1M+ transactions (2009-2011)

---

## ✨ Fonctionnalités principales

### 📊 **7 pages interactives**

1. **KPIs Overview** : 9 métriques clés + graphiques d'évolution CA
2. **Cohortes** : Heatmap de rétention, analyse LTV cumulée
3. **Segments RFM** : 7 segments clients (Champions, Loyaux, À risque, Perdus...)
4. **Scénarios** : Simulateur d'impact rétention/marge sur CLV + heatmap sensibilité
5. **Analyses Avancées** : Saisonnalité, affinité produits
6. **Export** : Listes CRM, données filtrées, visualisations PNG
7. **Qualité** : Rapport data quality et couverture

### 🔧 **Filtres dynamiques**
- Période temporelle
- Multi-pays
- Gestion des retours (3 modes)
- Seuil de commande

---

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/votre-username/marketing-analytics-dashboard.git
cd marketing-analytics-dashboard

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer le dashboard
streamlit run app/app.py
```

Le dashboard s'ouvre automatiquement à `http://localhost:8501`

---

## 📁 Structure

```
marketing-analytics-dashboard/
├── app
 └──app.py # Application Streamlit (2000+ lignes)
 └──app.py  # Fonctions de calcul (600+ lignes)                
├── requirements.txt          # Dépendances Python
├── README.md                 # Documentation
└── data/
    └── clean/
        └── online_retail_cleaned.csv
```

---

## 🧮 Méthodologie

### **Analyse de Cohortes**
Regroupe les clients par mois de première commande et suit leur rétention dans le temps.

### **Segmentation RFM**
Classe les clients selon 3 dimensions :
- **R**ecency : Jours depuis dernière commande
- **F**requency : Nombre de commandes
- **M**onetary : CA total généré

**7 segments** : Champions, Loyaux, Nouveaux, À risque, Hibernants, Potentiels, Perdus

### **Customer Lifetime Value (CLV)**

**3 méthodes de calcul :**

1. **CLV Empirique** : Somme du CA historique réel
   ```
   CLV = Σ(CA_mois_0 + CA_mois_1 + ... + CA_mois_11)
   ```

2. **CLV Formule** : Calcul théorique avec rétention
   ```
   CLV = m × r / (1 + d - r)
   ```
   - m = Marge mensuelle moyenne
   - r = Taux de rétention
   - d = Taux d'actualisation

3. **CLV Probabiliste** : Modèle BG/NBD (CLV personnalisée par client)

### **Simulation de Scénarios**
Teste l'impact de changements stratégiques (rétention, marge, remises) sur la CLV avec analyse de sensibilité interactive.

---

## 🛠️ Technologies

**Backend**
- Python 3.8+, Pandas, NumPy, SciPy

**Frontend**
- Streamlit, Plotly, Matplotlib, Seaborn

**Design**
- CSS personnalisé, gradient violet, Inter font

---

## 📋 Dépendances

```txt
pandas>=2.0.0
numpy>=1.24.0
streamlit>=1.28.0
plotly>=5.17.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
```

---

## 👤 Auteur

**Projet académique ECE Paris - DataViz 2025**

---

## 📄 License

MIT License - Voir [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- UCI Machine Learning Repository pour le dataset
- Streamlit & Plotly pour les frameworks
- Communauté Python open-source

---

<div align="center">


</div>


