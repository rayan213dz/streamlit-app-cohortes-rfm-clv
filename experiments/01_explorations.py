import pandas as pd
from pathlib import Path

# Chemins des fichiers
DATA_DIR = Path("../data/raw")  # à adapter si besoin
files = ["Cohortes.csv", "Cohortes2.csv"]

dfs = []
for f in files:
    path = DATA_DIR / f
    df_tmp = pd.read_csv(
        path,
        sep=";",          # séparateur ;
        decimal=",",      # virgule pour les prix
        encoding="utf-8"  # au cas où
    )
    dfs.append(df_tmp)

df = pd.concat(dfs, ignore_index=True)
df.head()
