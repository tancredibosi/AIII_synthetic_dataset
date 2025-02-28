import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer
from colorama import Fore, Style
from utils import *


def get_metadata(data):
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
    return metadata


def set_constraint(synthesizer):
    # load the constraint from the file
    synthesizer.load_custom_constraint_classes(
        filepath='custom_constraint_years.py',
        class_names=['CustomYearsHired']
    )
    years_hired_constraint = {
        'constraint_class': 'CustomYearsHired',
        'constraint_parameters': {
            'column_names': ['Candidate State', 'Year of insertion', 'Year of Recruitment']
        }
    }
    recruitment_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['Candidate State', 'Year of Recruitment']
        }
    }
    experience_age_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['Age Range', 'Years Experience']
        }
    }
    residence_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['City', 'Province', 'Region']
        }
    }
    event_constraint = {
        'constraint_class': 'FixedCombinations',
        'constraint_parameters': {
            'column_names': ['event_type__val', 'event_feedback']
        }
    }

    # Add constraint to the synthetizer
    synthesizer.add_constraints(constraints=[
        years_hired_constraint,
        recruitment_constraint,
        experience_age_constraint,
        residence_constraint,
        event_constraint
    ])
    return synthesizer


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

    data = cluster_tag(data)

    # Check inconsistencies
    inconsistencies_flag = data.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found: {inconsistencies_flag.sum()} inconsistencies {Style.RESET_ALL}")
    violating_rows = data[inconsistencies_flag]

    metadata = get_metadata(data)

    # Create syntetizer and generate new data
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        locales='it_IT',
        #    verbose=True
    )
    synthesizer.auto_assign_transformers(data)
    synthesizer.fit(data)
    synthetic_data = synthesizer.sample(num_rows=1000)

    # Check inconsistencies in synthetic data
    inconsistencies_flag = synthetic_data.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found: {inconsistencies_flag.sum()} inconsistencies {Style.RESET_ALL}")
    violating_rows = synthetic_data[inconsistencies_flag]

    synthesizer = set_constraint(synthesizer)

    # Generate data with constraint
    synthesizer.fit(data)
    synthetic_data_constraint = synthesizer.sample(num_rows=1000)
    # Check inconsistencies in synthetic data with constraint
    inconsistencies_flag = synthetic_data_constraint.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found: {inconsistencies_flag.sum()} inconsistencies {Style.RESET_ALL}")
    violating_rows = synthetic_data_constraint[inconsistencies_flag]

    # Generate polarized data
    print(f"\n{Fore.RED}Generating polarized data {Style.RESET_ALL}")

    polarization_list1 = [
        [{"Field": "Sex", "Value": "Female", "Percentage": 25}],
        [{"Field": "Candidate State", "Value": "Hired", "Percentage": 25}],
    ]

    final_data1, polarized_data1, remaining_synthetic_data1 = generate_polarized_data(synthesizer, polarization_list1, 1000, 3, 2)

    polarization_list2 = [
        [{"Field": "Sex", "Value": "Female", "Percentage": 25},
         {"Field": "Candidate State", "Value": "Hired", "Percentage": 25}],
        [{"Field": "Study Title", "Value": "Five-year degree", "Percentage": 10},
         {"Field": "Assumption Headquarters", "Value": "Milan", "Percentage": 10}]
    ]

    final_data2, polarized_data2, remaining_synthetic_data2 = generate_polarized_data(synthesizer, polarization_list2, 1000, 3, 2)
    print()
