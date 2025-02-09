import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import get_column_plot


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
    data = data.drop(columns=['linked_search__key', 'Years Experience.1', 'Study Area.1', 'Residence'])

    return data


def extract_number(value, choose_second=False):
    # Find all numbers in the string
    matches = re.findall(r'\d+', value)

    # If the flag is set to choose the second number
    if choose_second:
        if len(matches) > 1:
            return int(matches[1])  # Return the second number
        elif len(matches) > 0:
            return int(matches[0])  # Return the first if no second number
    # If the flag is False, always return the first number if it exists
    elif len(matches) > 0:
        return int(matches[0])
    return None  # Return None if no numbers are found


# True if at least one constraint violation is found
def check_constraint(row):
    flag = False
    flag += minor_worker_check(row)
    flag += hired_check(row)
    return bool(flag)


# Return True if the worker was a minor when started working
def minor_worker_check(row):
    age = extract_number(row['Age Range'], choose_second=True)
    work_exp = extract_number(row['Years Experience'])
    if work_exp == 'nan':
        return False
    return (age - work_exp) < 18


# If 'Candidate State'=='hired' -> 'Years of Recruitment' <= 'Years of Insertion'
# else 'Years of Recruitment' -> Nan
def hired_check(row):
    candidate_state = row['Candidate State']
    years_insert = row['Year of insertion']
    years_recruit = row['Year of Recruitment']

    if candidate_state == 'Hired':
        if pd.notna(years_insert) and pd.notna(years_recruit):
            if years_insert > years_recruit:
                return True
    else:
        if pd.notna(years_recruit):
            return True
    return False


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


def plot_distributions(data, synthetic_data, metadata, columns_to_plot):
    for col in columns_to_plot:
        fig = get_column_plot(
            real_data=data,
            synthetic_data=synthetic_data,
            metadata=metadata,
            column_name=col,
            plot_type='bar'
        )
        fig.show()


# For the moment can only generate one column at a time, like 20% Female and 30% Hired, can't generate 20% Hired Female
def polarized_generation(synthesizer, polarization_dict, num_rows=1000):
    synthetic_data = synthesizer.sample(num_rows=num_rows*5)

    reference_data = pd.DataFrame()
    for key, value in polarization_dict.items():
        n_elem = (num_rows * value['Percentage']) // 100
        fill_values = synthetic_data[synthetic_data[key] != value['Value']].head(num_rows-n_elem)
        polarized_rows = pd.DataFrame({key: [value['Value']] * n_elem})
        new_rows = pd.concat([polarized_rows, fill_values[key]]).reset_index(drop=True)
        reference_data = pd.concat([reference_data, new_rows], axis=1).reset_index(drop=True)

    polarized_synthetic_data = synthesizer.sample_remaining_columns(
        known_columns=reference_data,
        max_tries_per_batch=500
    )
    return polarized_synthetic_data
