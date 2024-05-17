import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.title("Data Preparations")
st.subheader("Raw Database of Fetivals in France")

data_url_raw = "/Users/mac/Desktop/festival_app_ara/data/festivals_global_festivals.csv"
data_url_clean = "/Users/mac/Desktop/festival_app_ara/data/df_clean_2.csv"


@st.cache_data
def load_data(df, sep=";"):
    df = pd.read_csv(df, sep=sep)
    return df


data_load_state = st.text("Loading Data ...")
data_raw = load_data(data_url_raw, sep=";")
data_clean = load_data(data_url_clean, sep=",")
data_load_state = st.text("Loading Fini!")

df = data_clean

if st.checkbox("Show Raw Database"):
    st.text("Festivals in France Data")
    st.dataframe(data_raw)


if st.checkbox("Show Clean Database"):
    st.text("Festivals in France Data")
    st.dataframe(data_clean)

# ....................................................................................................................
# Calculer le nombre de festivals par région
# Cela compte le nombre d'occurrences de chaque région et sélectionne les 10 premiers
festivals_par_region = df["Region"].value_counts().head(10).reset_index()

# Renommer les colonnes pour plus de clarté
festivals_par_region.columns = ["Region", "Nombre de festivals"]

# Créer un graphique à barres avec Plotly
fig = px.bar(
    festivals_par_region,
    x="Region",
    y="Nombre de festivals",
    title="Top 10 des régions avec le plus grand nombre de festivals",
    color_discrete_sequence=px.colors.sequential.thermal_r,
)

# Ajouter des étiquettes de texte au-dessus des barres
fig.update_traces(texttemplate="%{y}", textposition="outside")

# Afficher le graphique interactif
st.plotly_chart(fig)

# ....................................................................................................................
# Calculer le nombre total de festivals pour chaque année
festivals_par_annee = df["Annee"].value_counts().reset_index()
festivals_par_annee.columns = ["Annee", "Nombre de festivals"]

# Sélectionner les dix meilleures années
top_10_annees = festivals_par_annee.head(10)

# Calculer le nombre de festivals pour chaque combinaison 'Annee-Type'
festivals_counts = (
    df.groupby(["Annee", "Type"]).size().reset_index(name="Nombre de festivals")
)

# Filtrer les données pour ne conserver que les dix meilleures années
festivals_counts_top_10 = festivals_counts[
    festivals_counts["Annee"].isin(top_10_annees["Annee"])
]
festivals_counts_top_10 = festivals_counts_top_10.sort_values(
    by="Nombre de festivals", ascending=False
)


# Créer un graphique à barres avec Plotly en changeant la couleur des barres en fonction de la colonne 'Type'
fig = px.bar(
    festivals_counts_top_10,
    x="Annee",
    y="Nombre de festivals",
    title="Top 10 années avec le plus grand nombre de festivals en fonction des type",
    color="Type",  # Changer la couleur des barres en fonction de la colonne 'Type'
    color_discrete_sequence=px.colors.sequential.thermal_r, width= 1000
)

# Ajouter des étiquettes de texte au-dessus des barres
fig.update_traces(texttemplate="%{y}", textposition="outside")

# Afficher le graphique interactif
st.plotly_chart(fig)

# ....................................................................................................................
# Calculer le nombre total de festivals pour chaque année
festivals_par_annee = df["Annee"].value_counts().reset_index()
festivals_par_annee.columns = ["Annee", "Nombre de festivals"]

# Sélectionner les dix meilleures années
top_10_annees = festivals_par_annee.head(10)

# Calculer le nombre de festivals pour chaque combinaison 'Annee-Region-Type'
festivals_counts = (
    df.groupby(["Annee", "Region", "Type"])
    .size()
    .reset_index(name="Nombre de festivals")
)

# Filtrer les données pour ne conserver que les dix meilleures années
festivals_counts_top_10 = festivals_counts[
    festivals_counts["Annee"].isin(top_10_annees["Annee"])
]

# Calculer le nombre total de festivals pour chaque région
festivals_par_region = (
    festivals_counts_top_10.groupby("Region")["Nombre de festivals"].sum().reset_index()
)

# Sélectionner les cinq meilleures régions
top_5_regions = festivals_par_region.nlargest(5, "Nombre de festivals")["Region"]

# Filtrer les données pour ne conserver que les cinq meilleures régions
festivals_counts_top_10_top_5 = festivals_counts_top_10[
    festivals_counts_top_10["Region"].isin(top_5_regions)
]

# Créer un graphique à bulles catégorielles animé avec Plotly Express
fig = px.scatter(
    festivals_counts_top_10_top_5,
    x="Annee",
    y="Region",
    size="Nombre de festivals",
    title="Répartition des festivals par année et par région (Top 5)",
    color="Type",  # Changer la couleur des bulles en fonction de la colonne 'Type'
    size_max=100,  # Définir la taille maximale des bulles
    animation_frame="Annee", width= 1300
)  # Animation basée sur l'année

# Afficher le graphique interactif
st.plotly_chart(fig)
