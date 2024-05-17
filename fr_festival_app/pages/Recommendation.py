import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import folium
from streamlit_folium import folium_static

# Initialiser les stopwords
nltk.download("stopwords")
stop_word = stopwords.words("french")
stop_word += stopwords.words("english")

# Charger le dataset
df = pd.read_csv(
    "/Users/mac/Desktop/festival_app_ara/fr_festival_app/pages/Ordered_NLP_preprocessed_df.csv"
)

# Renommer les colonne pour être plus parlant
nouveaux_noms = {
    "Processed_nom_festival": "Nom_festival",
    "Processed_Type": "Type",
    "Processed_Region": "Region",
    "Processed_Ville": "Ville",
}
df.rename(columns=nouveaux_noms, inplace=True)


# Définir les fonctions de prétraitement
def tok(sentence):
    return nltk.word_tokenize(sentence.lower())


def no_stop(tokens):
    return [token for token in tokens if (token not in stop_word)]


def stem(tokens, language="french"):
    stemmizer = SnowballStemmer(language=language)
    return [stemmizer.stem(token) for token in tokens]


def lemmatize(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess(sentence, stemm=True, lemm=True, stop=True):
    tokens = tok(sentence)
    if stop:
        tokens = no_stop(tokens)
    if lemm:
        tokens = lemmatize(tokens)
    if stemm and not lemm:  # On ne fait pas le stemming si la lemmatisation est activée
        tokens = stem(tokens)
    return " ".join(tokens)


# Nettoyaer les colonnes et créer la colonne 'Soop' qui contient les parametres souhaités pour notre entrainement


# Fonction pour nettoyer les valeurs en excluant les chiffres et les virgules
def clean_value(value):
    if pd.isna(value):  # Vérifie si la valeur est NaN
        return ""
    cleaned_value = re.sub(
        r"[\d,]", "", str(value)
    )  # Supprime les chiffres et les virgules
    return cleaned_value.strip()  # Supprime les espaces en début et fin de chaîne


# Colonnes à utiliser pour créer la colonne 'Soop'
colonnes_utilisees = [
    "Nom_festival",
    "Type",
    "Region",
    "Ville",
    "Procced_musique",
    "Processed_Spectacle_vivant",
    "Processed_Cinema_audiovisuel",
    "Processed_Livre_litterature",
]

# Créer des colonnes temporaires nettoyées
for col in colonnes_utilisees:
    df[f"{col}_clean"] = df[col].apply(clean_value)

# Créer la nouvelle colonne 'Soop' en combinant les valeurs des colonnes temporaires nettoyées
df["Soop"] = df[[f"{col}_clean" for col in colonnes_utilisees]].apply(
    lambda row: " ".join(row.values), axis=1
)

# Supprimer les colonnes temporaires nettoyées
df.drop(columns=[f"{col}_clean" for col in colonnes_utilisees], inplace=True)


# Prétraiter les données
df["Cleaned_Soop"] = df["Soop"].apply(lambda x: preprocess(x, stemm=True, lemm=False))

# Initialisation des vectorizers et transformation des données
tfidf_vectorizer = TfidfVectorizer()
X = df["Cleaned_Soop"]
tfidf_vectorizer.fit(X)
X_tfidf = tfidf_vectorizer.transform(X)

# Initialisation et entraînement du modèle KNN
modelNN_t = NearestNeighbors(n_neighbors=3, metric="cosine")
modelNN_t.fit(X_tfidf)


# Définir la fonction pour trouver les entrées les plus proches
def find_closest_entries(query, vectorizer, knn_model, df, n_neighbors=5):
    # Prétraiter la requête
    query_cleaned = preprocess(query, stemm=True, lemm=False)

    # Transformer la requête en TF-IDF
    query_tfidf = vectorizer.transform([query_cleaned]).toarray()

    # Trouver les entrées les plus proches
    distances, indices = knn_model.kneighbors(query_tfidf, n_neighbors=n_neighbors)

    # Extraire les entrées similaires
    # Extraire les entrées similaires

    similar_entries = df.loc[
        indices[0],
        [
            "Nom_festival",
            "Type",
            "Region",
            "Ville",
            "Annee",
            "Geocode",
            "Site_internet",
        ],
    ]
    return similar_entries


# Créer une carte centrée sur la France
def create_map(similar_entries):
    carte = folium.Map(location=[46.603354, 1.888334], zoom_start=6)

    # Ajouter des marqueurs pour chaque position géographique dans les entrées similaires
    for index, row in similar_entries.iterrows():
        geocode = row["Geocode"].split(",")
        if (
            len(geocode) == 2
        ):  # Vérifiez que vous avez exactement deux parties après split
            lat = float(geocode[0].strip())
            lon = float(geocode[1].strip())
            nom_festival = row["Nom_festival"]
            folium.Marker(location=[lat, lon], tooltip=nom_festival).add_to(carte)

    return carte


# Interface utilisateur Streamlit
st.title("Recherche de festivale")

# Utiliser une barre latérale pour la recherche
# with st.sidebar:
query = st.text_input("Entrez une description du festival :", "")

if query:
    # Trouver les entrées similaires
    closest_entries = find_closest_entries(
        query, tfidf_vectorizer, modelNN_t, df, n_neighbors=5
    )

    # Afficher les résultats
    st.write("Les festivales similaires :")
    st.dataframe(closest_entries)

    # Afficher la carte avec les positions des festivals similaires
    carte = create_map(closest_entries)
    folium_static(carte)

    # Afficher la prévisualisation des sites web sous forme de liens
    st.write(
        "Cliquez sur le lien ci-dessous pour visiter le site de l'organisation festival :"
    )
    for index, row in closest_entries.iterrows():
        nom_festival = row["Nom_festival"]
        site = row["Site_internet"]
        if pd.notna(site):
            # Ajouter 'http://' si l'URL ne commence pas par 'http' ou 'https'
            if not site.startswith(("http://", "https://")):
                site = "http://" + site
            st.markdown(f"[{nom_festival}]({site})")
