import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter


def organize_data(data):
    data.columns = data.columns.str.strip()
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop duplicate rows
    data = data.drop_duplicates()

    # Drop the tilde in the 'Overall' column
    data['Overall'] = data['Overall'].str.lstrip('~ ')

    # Extract 'City', 'Province' and 'Region' from the column 'Residence'
    data[['City', 'Province', 'Region']] = data['Residence'].str.split(' » | ~ ', expand=True)

    # Convert the columns 'Year of insertion' and 'Year of Recruitment' to integers
    data['Year of insertion'] = pd.to_numeric(data['Year of insertion'].str.strip('[]'), errors='coerce').astype(
        'Int64')
    data['Year of Recruitment'] = pd.to_numeric(data['Year of Recruitment'].str.strip('[]'), errors='coerce').astype(
        'Int64')

    undesired_values = ['????', '-', '.', '/']
    data.loc[data['Last Role'].isin(undesired_values), 'Last Role'] = np.nan

    # Group the same IDs in a unique row
    def group_ids(df):
        # Count non-NaN values in each row
        df['non_nan_count'] = df.notna().sum(axis=1)
        # Keep the row with the highest non-NaN count per ID
        df = df.loc[df.groupby('ID')['non_nan_count'].idxmax()]
        # Drop helper column
        df = df.drop(columns=['non_nan_count'])
        return df

    data = group_ids(data)

    # Drop useless columns
    data = data.drop(columns=['linked_search__key', 'Years Experience.1', 'Study Area.1', 'Residence', 'Recruitment Request'])

    return data


def preprocess_text(text):
    """Rimuove caratteri speciali e converte il testo in minuscolo."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Rimuove caratteri speciali
    return text.lower()


def cluster_and_map_roles(unique_values):
    processed_values = [preprocess_text(value) for value in unique_values]

    # Creazione della matrice TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_values)

    # Clustering gerarchico
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='ward')
    clusters = clustering.fit_predict(X.toarray())

    # Creazione del dizionario con i cluster
    value_clusters = {}
    for value, cluster_id in zip(unique_values, clusters):
        if cluster_id not in value_clusters:
            value_clusters[cluster_id] = []
        value_clusters[cluster_id].append(value)

    # Assegna un nome a ogni cluster basato sulle parole più comuni
    cluster_names = {}
    for cluster_id, values in value_clusters.items():
        words = []
        for value in values:
            words.extend(preprocess_text(value).split())
        common_words = [word for word, count in Counter(words).most_common(2)]
        cluster_names[cluster_id] = "-".join(common_words) if common_words else "Unknown"

    return clusters, cluster_names


def map_to_cluster_name(value, unique_values, clusters, cluster_names):
    if value in unique_values:
        cluster_id = clusters[unique_values.index(value)]
        return cluster_names.get(cluster_id, value)
    return value


def cluster_tag(data):
    # Supponiamo che unique_last_roles e unique_tag siano liste estratte dai dati
    unique_last_roles = data['Last Role'].dropna().unique().tolist()
    unique_tag = data['TAG'].dropna().unique().tolist()
    # Clusterizzazione e assegnazione dei nomi per Last Role
    clusters_last_roles, cluster_names_last_roles = cluster_and_map_roles(unique_last_roles)
    clusters_tags, cluster_names_tags = cluster_and_map_roles(unique_tag)
    data['Last Role'] = data['Last Role'].apply(
        lambda role: map_to_cluster_name(role, unique_last_roles, clusters_last_roles, cluster_names_last_roles))
    data['TAG'] = data['TAG'].apply(lambda tag: map_to_cluster_name(tag, unique_tag, clusters_tags, cluster_names_tags))
    return data