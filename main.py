import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, CopulaGANSynthesizer
from colorama import Fore, Style
from utils import *

if __name__ == '__main__':
    # Import data and clean
    file_path = 'Dataset_2.0_Akkodis.xlsx'
    # Import the dataset into a pandas DataFrame
    original_data = pd.read_excel(file_path)
    data = organize_data(original_data)

    # Clean data from inconsistencies
    invalid_mask = (
            ((data['Age Range'] == '< 20 years') &
             (data['Years Experience'].isin(['[+10]', '[7-10]', '[5-7]', '[3-5]']))) |
            ((data['Age Range'] == '20 - 25 years') &
             (data['Years Experience'] == '[+10]'))
    )
    # Remove invalid rows
    data = data[~invalid_mask].copy()

    # Supponiamo che unique_last_roles e unique_tag siano liste estratte dai dati
    unique_last_roles = data['Last Role'].dropna().unique().tolist()
    unique_tag = data['TAG'].dropna().unique().tolist()
    # Clusterizzazione e assegnazione dei nomi per Last Role
    clusters_last_roles, cluster_names_last_roles = cluster_and_map_roles(unique_last_roles)
    clusters_tags, cluster_names_tags = cluster_and_map_roles(unique_tag)
    data['Last Role'] = data['Last Role'].apply(
        lambda role: map_to_cluster_name(role, unique_last_roles, clusters_last_roles, cluster_names_last_roles))
    data['TAG'] = data['TAG'].apply(lambda tag: map_to_cluster_name(tag, unique_tag, clusters_tags, cluster_names_tags))

    # Check inconsistencies
    inconsistencies_flag = data.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found: {inconsistencies_flag.sum()} inconsistencies {Style.RESET_ALL}")
    violating_rows = data[inconsistencies_flag]

    # Set up metadata
    metadata = Metadata.detect_from_dataframe(data=data)
    # Set column Candidate State to categorical
    metadata.update_column(
        column_name='Candidate State',
        sdtype='categorical')
    # Set column Last Role to categorical
    metadata.update_column(
        column_name='Last Role',
        sdtype='categorical')
    metadata.update_column(
        column_name='City',
        sdtype='categorical')
    metadata.update_column(
        column_name='Province',
        sdtype='categorical')
    metadata.update_column(
        column_name='Region',
        sdtype='categorical')
    metadata.validate()

    # Create syntetizer and generate new data
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        locales='it_IT',
        #    verbose=True
    )
    """synthesizer.auto_assign_transformers(data)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows=1000)

    # Check inconsistencies in synthetic data
    inconsistencies_flag = synthetic_data.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found: {inconsistencies_flag.sum()} inconsistencies {Style.RESET_ALL}")
    violating_rows = synthetic_data[inconsistencies_flag]"""

    # load the constraint from the file
    synthesizer.load_custom_constraint_classes(
        filepath='custom_constraint_years.py',
        class_names=['CustomYearsHired', 'CustomAgeExperience']
    )
    YearsHired_constraint = {
        'constraint_class': 'CustomYearsHired',
        'constraint_parameters': {
            'column_names': ['Candidate State', 'Year of insertion', 'Year of Recruitment']
        }
    }
    AgeExperience_constraint = {
        'constraint_class': 'CustomAgeExperience',
        'constraint_parameters': {
            'column_names': ['Age Range', 'Years Experience']
        }
    }
    experience_age_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['Age Range', 'Years Experience']
        }
    }
    recruitment_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['Candidate State', 'Year of Recruitment']
        }
    }
    residence_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['City', 'Province', 'Region']
        }
    }

    # Add constraint to the synthetizer
    synthesizer.add_constraints(constraints=[
        experience_age_constraint,
        recruitment_constraint,
        residence_constraint,
        YearsHired_constraint,
        AgeExperience_constraint
    ])

    # Generate data with constraint
    synthesizer.fit(data)
    synthetic_data_constraint = synthesizer.sample(num_rows=1000)

    # Check inconsistencies in synthetic data with constraint
    inconsistencies_flag = synthetic_data_constraint.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found: {inconsistencies_flag.sum()} inconsistencies {Style.RESET_ALL}")
    violating_rows = synthetic_data_constraint[inconsistencies_flag]

    print()