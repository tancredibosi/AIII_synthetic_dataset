import pandas as pd
import numpy as np
import re
from colorama import Fore, Style
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import matplotlib.pyplot as plt
from sdv.evaluation.single_table import get_column_plot
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.sampling import Condition
import math
from collections import defaultdict
from colorama import Fore, Style


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


def polarized_generation_from_conditions(synthesizer, polarization_list, num_rows=1000, scaling_factor=2, max_retries=3):
    synthetic_data_list = []

    retries = 0
    while retries < max_retries:
        try:
            tot_rows = 0
            for sublist in polarization_list:
                n_elem = (num_rows * sublist[0]['Percentage']) // 100
                tot_rows += n_elem
                prev_percentage = None
                col_values = {}
                for el in sublist:
                    if prev_percentage and prev_percentage != el['Percentage']:
                        raise ValueError("Error: Mismatched percentages")
                    col_values[el['Field']] = el['Value']

                condition = Condition(
                    num_rows=n_elem * scaling_factor,
                    column_values=col_values
                )
                polarized_synthetic_data = synthesizer.sample_from_conditions(
                    conditions=[condition],
                )
                synthetic_data_list.append(polarized_synthetic_data)

            # Combine all generated synthetic data
            all_polarized_data = pd.concat(synthetic_data_list).reset_index(drop=True)
            filtered_polarized_data = filter_by_constraints(all_polarized_data, polarization_list, num_rows)
            break
        except Exception as e:
            retries += 1
            scaling_factor *= 2  # Increase scaling factor on retry
            print(f"Retry {retries}/{max_retries}: Increasing scaling factor to {scaling_factor} due to error: {e}")
            if retries == max_retries:
                raise RuntimeError("Max retries reached. Unable to generate valid polarized synthetic data.")

    scaling_factor = 3
    retries = 0
    while retries < max_retries:
        try:
            synthetic_data = synthesizer.sample(num_rows=num_rows * scaling_factor)
            filtered_synthetic_data = filter_dataframe(synthetic_data, polarization_list)
            fill_values = filtered_synthetic_data.sample(n=num_rows-tot_rows)
            break
        except Exception as e:
            retries += 1
            scaling_factor *= 2  # Increase scaling factor on retry
            print(f"Retry {retries}/{max_retries}: Increasing scaling factor to {scaling_factor} due to error: {e}")
            if retries == max_retries:
                raise RuntimeError("Max retries reached. Unable to generate valid polarized synthetic data.")

    final_data = pd.concat([filtered_polarized_data, fill_values]).reset_index(drop=True)
    return final_data


def filter_by_constraints(df, polarization_list, num_rows):
    final_df = pd.DataFrame()  # Create an empty DataFrame to store results
    used_indices = set()  # To track already used rows and avoid conflicts

    for i, constraint_group in enumerate(polarization_list):
        temp_df = df.copy()

        # Exclude rows that match constraints from other groups
        for j, other_group in enumerate(polarization_list):
            if i != j:  # Only consider other groups
                for constraint in other_group:
                    field, value = constraint["Field"], constraint["Value"]
                    temp_df = temp_df[temp_df[field] != value]  # Exclude matching rows

        # Now apply the current constraint group
        for constraint in constraint_group:
            field, value = constraint["Field"], constraint["Value"]
            num_elem = (num_rows * constraint['Percentage']) // 100
            temp_df = temp_df[temp_df[field] == value]  # Apply filter

        # Remove already used rows to prevent conflicts
        temp_df = temp_df.loc[~temp_df.index.isin(used_indices)]

        # Ensure the result matches the required Num_elem
        if len(temp_df) >= num_elem:
            temp_df = temp_df.head(num_elem)  # Take only required rows
        else:
            raise ValueError(f"Not enough rows for constraints {constraint_group}, missing {num_elem - len(temp_df)} elements")

        # Update used indices to avoid conflicts
        used_indices.update(temp_df.index)

        # Append results to final DataFrame
        final_df = pd.concat([final_df, temp_df])

    return final_df.reset_index(drop=True)


def polarized_generation_from_columns(synthesizer, polarization_list, num_rows=1000):
    synthetic_data = synthesizer.sample(num_rows=num_rows*5)

    reference_data = pd.DataFrame()
    for sublist in polarization_list:
        prev_percentage = None
        exclude_conditions = {}
        new_rows = pd.DataFrame()
        for el in sublist:
            if prev_percentage and prev_percentage != el['Percentage']:
                raise ValueError("Error: Mismatched percentages")

            exclude_conditions[el['Field']] = el['Value']
            n_elem = (num_rows * el['Percentage']) // 100
            polarized_rows = pd.DataFrame({el['Field']: [el['Value']] * n_elem})
            new_rows = pd.concat([new_rows, polarized_rows], axis=1).reset_index(drop=True)
            prev_percentage = el['Percentage']

        mask = ~synthetic_data.apply(lambda row: all(row[col] == val for col, val in exclude_conditions.items()), axis=1)
        fill_values = synthetic_data[mask].sample(n=num_rows-n_elem)
        new_rows = pd.concat([new_rows, fill_values[new_rows.columns]]).reset_index(drop=True)
        reference_data = pd.concat([reference_data, new_rows], axis=1).reset_index(drop=True)
        reference_data = reference_data.sample(frac=1).reset_index(drop=True)

    polarized_synthetic_data = synthesizer.sample_remaining_columns(
        known_columns=reference_data,
        max_tries_per_batch=500,
        batch_size=1024,
    )
    return polarized_synthetic_data


def check_distribution_constraints(df, constraints_list):
    total_rows = len(df)
    if total_rows == 0:
        return False

    for constraint_group in constraints_list:
        filtered_df = df.copy()

        for condition in constraint_group:
            field, value = condition["Field"], condition["Value"]
            filtered_df = filtered_df[filtered_df[field] == value]

        actual_count = len(filtered_df)
        actual_percentage = (actual_count / total_rows) * 100
        expected_percentage = constraint_group[0]["Percentage"]
        expected_count = round((expected_percentage / 100) * total_rows)

        if round(actual_percentage, 2) != round(expected_percentage, 2):
            print(f"{Fore.RED}Distribution of polarization not respected{Style.RESET_ALL}")
            print(f"Condition: {constraint_group}")
            print(f"Actual Count: {actual_count}, Expected Count: {expected_count}")
            print(f"Actual Percentage: {actual_percentage}%, Expected Percentage: {expected_percentage}%")
            return False  # Constraint is not met

    print(f"{Fore.GREEN}Distribution of polarization respected{Style.RESET_ALL}")
    return True  # All constraints are satisfied


def filter_dataframe(df, polarization_list):
    # Create an initial mask of all False (i.e., no rows are removed initially)
    mask_to_remove = pd.Series(False, index=df.index)

    for condition_set in polarization_list:
        # Start with a mask of all True for the given condition set
        condition_mask = pd.Series(True, index=df.index)

        for condition in condition_set:
            field, value = condition["Field"], condition["Value"]
            condition_mask &= (df[field] == value)

        # Accumulate rows to remove using OR
        mask_to_remove |= condition_mask

    # Filter out rows that match the conditions
    return df[~mask_to_remove]


def plot_comparison_subplots(drs1_dict, drs2_dict, dqs1_dict, dqs2_dict, 
                             title1="Diagnostic Scores Comparison", 
                             title2="Quality Scores Comparison", 
                             dict1_name="Synthesizer 1", 
                             dict2_name="Synthesizer 2"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Diagnostic Scores
    labels1 = list(drs1_dict.keys())
    drs1_values = list(drs1_dict.values())
    drs2_values = list(drs2_dict.values())
    x1 = np.arange(len(labels1))
    width = 0.35
    
    rects1a = axes[0].bar(x1 - width/2, drs1_values, width, label=dict1_name)
    rects1b = axes[0].bar(x1 + width/2, drs2_values, width, label=dict2_name)
    
    axes[0].set_ylabel('Scores')
    axes[0].set_title(title1)
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(labels1, rotation=45, ha='right')
    axes[0].legend()
    
    # Plot Quality Scores
    labels2 = list(dqs1_dict.keys())
    dqs1_values = list(dqs1_dict.values())
    dqs2_values = list(dqs2_dict.values())
    x2 = np.arange(len(labels2))
    
    rects2a = axes[1].bar(x2 - width/2, dqs1_values, width, label=dict1_name)
    rects2b = axes[1].bar(x2 + width/2, dqs2_values, width, label=dict2_name)
    
    axes[1].set_ylabel('Scores')
    axes[1].set_title(title2)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2, rotation=45, ha='right')
    axes[1].legend()
    
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1a, axes[0])
    autolabel(rects1b, axes[0])
    autolabel(rects2a, axes[1])
    autolabel(rects2b, axes[1])
    
    plt.tight_layout()
    plt.show()


def compare_synthesizer(s1, s2, metadata, data, num_rows=1000):
    s1.fit(data)
    s2.fit(data)
    
    synthetic_data_s1 = s1.sample(num_rows=num_rows)
    synthetic_data_s2 = s2.sample(num_rows=num_rows)
    
    diagnostic_report_s1 = run_diagnostic(data, synthetic_data_s1, metadata, verbose=False)
    drs1_dict = dict(zip(diagnostic_report_s1.get_properties()['Property'], diagnostic_report_s1.get_properties()['Score']))
    
    quality_report_s1 = evaluate_quality(data, synthetic_data_s1, metadata, verbose=False)
    dqs1_dict = dict(zip(quality_report_s1.get_properties()['Property'], quality_report_s1.get_properties()['Score']))
    
    diagnostic_report_s2 = run_diagnostic(data, synthetic_data_s2, metadata, verbose=False)
    drs2_dict = dict(zip(diagnostic_report_s2.get_properties()['Property'], diagnostic_report_s2.get_properties()['Score']))
    
    quality_report_s2 = evaluate_quality(data, synthetic_data_s2, metadata, verbose=False)
    dqs2_dict = dict(zip(quality_report_s2.get_properties()['Property'], quality_report_s2.get_properties()['Score']))
    
    plot_comparison_subplots(drs1_dict, drs2_dict, dqs1_dict, dqs2_dict, 
                             dict1_name=f"{s1.__class__.__name__}",
                             dict2_name=f"{s2.__class__.__name__}")


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


def process_data_polarization(synthesizer, polarization_list, num_rows, scaling_factor, attempt):
    """
    Generate synthetic data and apply polarization rules.
    """
    current_multiplier = 100 * (scaling_factor ** (attempt - 1))
    total_rows = num_rows * current_multiplier

    print(f"\nTentativo {attempt}: Generazione di {total_rows} righe...")
    initial_data = synthesizer.sample(num_rows=total_rows)

    polarized_data = []
    used_values = defaultdict(set)
    total_polarized_rows = 0

    for sublist in polarization_list:
        subset_conditions = {el['Field']: el['Value'] for el in sublist}
        percentage = sublist[0]['Percentage']
        n_elem = math.floor((num_rows * percentage) / 100)
        total_polarized_rows += n_elem

        condition_df = initial_data.copy()

        # Exclude data belonging to other polarization groups
        for excluded_sublist in polarization_list:
            if excluded_sublist != sublist:
                for el in excluded_sublist:
                    condition_df = condition_df[condition_df[el['Field']] != el['Value']]

        # Apply filtering conditions
        for field, value in subset_conditions.items():
            condition_df = condition_df[condition_df[field] == value]

        print(f"Condizioni: {subset_conditions} - Righe richieste: {n_elem} - Disponibili: {len(condition_df)}")

        if len(condition_df) < n_elem:
            raise ValueError(f"Dati insufficienti per {subset_conditions}. Disponibili: {len(condition_df)}, richiesti: {n_elem}.")

        selected_data = condition_df.sample(n=n_elem, replace=False)
        polarized_data.append(selected_data)

        for field, value in subset_conditions.items():
            used_values[field].add(value)

    polarized_synthetic_data = pd.concat(polarized_data, ignore_index=True)
    return initial_data, polarized_synthetic_data, used_values, total_polarized_rows


def generate_polarized_data(synthesizer, polarization_list, num_rows=1000, max_retries=3, scaling_factor=2):
    """
    Generate a synthetic dataset with polarized data and fallback logic.
    """
    for attempt in range(1, max_retries + 1):
        try:
            initial_data, polarized_data, used_values, total_polarized_rows = process_data_polarization(
                synthesizer, polarization_list, num_rows, scaling_factor, attempt
            )

            remaining_data = initial_data.copy()
            for field, values in used_values.items():
                remaining_data = remaining_data[~remaining_data[field].isin(values)]

            num_remaining_rows = num_rows - total_polarized_rows
            remaining_synthetic_data = remaining_data.sample(n=num_remaining_rows, replace=False) if num_remaining_rows > 0 else pd.DataFrame()

            final_data = pd.concat([polarized_data, remaining_synthetic_data]).sample(frac=1).reset_index(drop=True)

            print(f"Dataset finale generato: {len(final_data)} righe (attese: {num_rows})")
            return final_data, polarized_data, remaining_synthetic_data

        except ValueError as e:
            if attempt < max_retries:
                print(f"{Fore.YELLOW}Errore: {str(e)}. Nuovo tentativo...{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Errore dopo {max_retries} tentativi: {str(e)}{Style.RESET_ALL}")
                raise