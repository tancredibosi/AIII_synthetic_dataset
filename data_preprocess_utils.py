import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter


def organize_data(data):
    """
    Cleans and organizes the dataset.
    - Strips whitespace from column names and string values.
    - Removes duplicate rows.
    - Cleans the 'Overall' column and splits 'Residence' column into separate columns.
    - Converts 'Year of insertion' and 'Year of Recruitment' columns to integers.
    - Filters out rows with invalid 'Last Role' values.
    - Groups rows with the same ID by keeping the one with the most non-NaN values.
    - Drops unnecessary columns.
    """
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces from column names
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)  # Strip string values in all cells

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

    # Remove invalid values in 'Last Role' column
    undesired_values = ['????', '-', '.', '/']
    data.loc[data['Last Role'].isin(undesired_values), 'Last Role'] = np.nan

    # Group rows by ID, keeping the one with the most non-NaN values
    def group_ids(df):
        """
        Groups the dataset by ID, keeping the row with the highest number of non-NaN values for each ID.
        """
        # Count non-NaN values in each row
        df['non_nan_count'] = df.notna().sum(axis=1)
        # Keep the row with the highest non-NaN count per ID
        df = df.loc[df.groupby('ID')['non_nan_count'].idxmax()]
        # Drop the helper column 'non_nan_count'
        df = df.drop(columns=['non_nan_count'])
        return df

    data = group_ids(data)

    # Drop columns that are not useful for the analysis
    data = data.drop(columns=['linked_search__key', 'Years Experience.1', 'Study Area.1', 'Residence', 'Recruitment Request'])

    return data


def preprocess_text(text):
    """
    Preprocesses text by removing special characters and converting it to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z ]', '', text)
    # Convert to lowercase
    return text.lower()


def filter_minor_workers(data):
    """
    Filters out workers who have inconsistencies in their age range and years of experience.
    Specifically, removes rows with invalid combinations of 'Age Range' and 'Years Experience'.
    """
    # Clean data from inconsistencies based on age and experience
    invalid_mask = (
            ((data['Age Range'] == '< 20 years') &
             (data['Years Experience'].isin(['[+10]', '[7-10]', '[5-7]', '[3-5]']))) |
            ((data['Age Range'] == '20 - 25 years') &
             (data['Years Experience'] == '[+10]'))
    )
    # Remove invalid rows
    return data[~invalid_mask].copy()


def cluster_and_map_roles(unique_values):
    """
    Clusters a list of unique values (e.g., job roles or tags) using hierarchical clustering.
    Returns the cluster assignments and cluster names based on common words.
    """
    # Preprocess the unique values (convert to lowercase and remove special characters)
    processed_values = [preprocess_text(value) for value in unique_values]

    # Create the TF-IDF matrix
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_values)

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='ward')
    clusters = clustering.fit_predict(X.toarray())

    # Create a dictionary that maps cluster IDs to the corresponding values
    value_clusters = {}
    for value, cluster_id in zip(unique_values, clusters):
        if cluster_id not in value_clusters:
            value_clusters[cluster_id] = []
        value_clusters[cluster_id].append(value)

    # Assign a name to each cluster based on the most common words in the values
    cluster_names = {}
    for cluster_id, values in value_clusters.items():
        words = []
        for value in values:
            words.extend(preprocess_text(value).split())
        common_words = [word for word, count in Counter(words).most_common(2)]
        cluster_names[cluster_id] = "-".join(common_words) if common_words else "Unknown"

    return clusters, cluster_names


def map_to_cluster_name(value, unique_values, clusters, cluster_names):
    """
    Maps a value to its corresponding cluster name.
    """
    if value in unique_values:
        # Find the cluster ID for the value
        cluster_id = clusters[unique_values.index(value)]
        # Return the corresponding cluster name, or the value itself if not found
        return cluster_names.get(cluster_id, value)
    return value


def cluster_tag(data):
    """
    Applies clustering to 'Last Role' and 'TAG' columns in the dataset, assigning each value to a cluster name.
    """
    # Extract unique values for 'Last Role' and 'TAG' columns
    unique_last_roles = data['Last Role'].dropna().unique().tolist()
    unique_tag = data['TAG'].dropna().unique().tolist()

    # Perform clustering and assign names to clusters for 'Last Role' and 'TAG'
    clusters_last_roles, cluster_names_last_roles = cluster_and_map_roles(unique_last_roles)
    clusters_tags, cluster_names_tags = cluster_and_map_roles(unique_tag)

    # Map the original 'Last Role' and 'TAG' values to their cluster names
    data['Last Role'] = data['Last Role'].apply(
        lambda role: map_to_cluster_name(role, unique_last_roles, clusters_last_roles, cluster_names_last_roles))
    data['TAG'] = data['TAG'].apply(lambda tag: map_to_cluster_name(tag, unique_tag, clusters_tags, cluster_names_tags))

    return data
