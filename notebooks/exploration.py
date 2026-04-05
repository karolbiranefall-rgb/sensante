import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement des données
# On utilise '../' pour remonter d'un niveau (sortir de notebooks/ pour aller dans data/)
try:
    df = pd.read_csv('../data/patients_dakar.csv')
    print("✅ Fichier chargé avec succès !")
except FileNotFoundError:
    print("❌ Erreur : Le fichier CSV est introuvable dans le dossier data/")

# 2. Aperçu des 5 premières lignes
print("\n--- Aperçu des données ---")
print(df.head())

# 3. Informations sur les colonnes et types
print("\n--- Infos Dataset ---")
print(df.info())

# 4. Analyse des diagnostics (Paludisme, Grippe, etc.)
print("\n--- Répartition des maladies ---")
print(df['diagnostic'].value_counts())

# 5. Visualisation simple
df['diagnostic'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Répartition des diagnostics - SénSanté')
plt.show()