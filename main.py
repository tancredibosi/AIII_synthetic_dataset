import warnings
import pandas as pd
import time
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from utils import *
from data_plots_utils import *
from data_preprocess_utils import *
from data_check_utils import *

# Suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Function to detect and set metadata for columns
def get_metadata(data):
    """
    Detects and sets metadata for the input data.
    Specifically, it updates certain columns as categorical.
    """
    # Detect metadata from the DataFrame
    metadata = Metadata.detect_from_dataframe(data=data)

    # List of columns to be marked as categorical
    categorical_columns = ['Candidate State', 'Last Role', 'City', 'Province', 'Region']

    # Update specified columns to be categorical
    for column in categorical_columns:
        metadata.update_column(column_name=column, sdtype='categorical')

    # Validate the metadata
    metadata.validate()
    return metadata


# Function to load and set constraints for the synthesizer
def set_constraints(synthesizer):
    """
    Loads custom constraints and adds them to the synthesizer.
    """
    # Load the custom constraint classes
    synthesizer.load_custom_constraint_classes(
        filepath='custom_constraint_years.py',
        class_names=['CustomYearsHired']
    )

    # Define constraints
    constraints = [
        # Custom constraint for years hired
        {
            'constraint_class': 'CustomYearsHired',
            'constraint_parameters': {'column_names': ['Candidate State', 'Year of insertion', 'Year of Recruitment']}
        },
        # Fixed combination constraints
        {
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['Candidate State', 'Year of Recruitment']}
        },
        {
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['Age Range', 'Years Experience']}
        },
        {
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['City', 'Province', 'Region']}
        },
        {
            'constraint_class': 'FixedCombinations',
            'constraint_parameters': {'column_names': ['event_type__val', 'event_feedback']}
        }
    ]

    # Add constraints to the synthesizer
    synthesizer.add_constraints(constraints=constraints)
    return synthesizer


"""
To discuss:
- How data cleaning? Why that data cleaning?
- How Synthesizer works? Why that Synthesizer? -> Show comparision between time to train and performance of the 3 synthesizer
- How applied constraint? Why need for a Custom one?
- How polarized data? Why need to use custom function? Comparison of the two methods performance
"""

# Main execution flow
if __name__ == '__main__':
    # Load and clean data
    file_path = 'Dataset_2.0_Akkodis.xlsx'
    original_data = pd.read_excel(file_path)

    # Apply preprocessing steps
    data = organize_data(original_data)
    data = filter_minor_workers(data)
    data = cluster_tag(data)

    # Check for data inconsistencies
    inconsistencies_flag = data.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found {inconsistencies_flag.sum()} inconsistencies in the original data{Style.RESET_ALL}")
    time.sleep(1)

    # Display rows with violations
    violating_rows = data[inconsistencies_flag]

    # Initialize synthesizer
    metadata = get_metadata(data)
    synthesizer = GaussianCopulaSynthesizer(metadata, locales='it_IT')

    # Auto-assign transformers and fit the synthesizer to the data
    synthesizer.auto_assign_transformers(data)
    synthesizer.fit(data)

    # Generate synthetic data without constraints
    synthetic_data = synthesizer.sample(num_rows=1000)
    time.sleep(1)

    # Check for inconsistencies in the synthetic data
    inconsistencies_flag = synthetic_data.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found {inconsistencies_flag.sum()} inconsistencies in the synthetic data{Style.RESET_ALL}")
    time.sleep(1)

    # Display rows with violations
    violating_rows = synthetic_data[inconsistencies_flag]

    # Apply constraints to the synthesizer
    synthesizer = set_constraints(synthesizer)

    # Generate synthetic data with constraints
    synthesizer.fit(data)
    synthetic_data_with_constraints = synthesizer.sample(num_rows=1000)

    # Check for inconsistencies in the synthetic data with constraints
    inconsistencies_flag = synthetic_data_with_constraints.apply(check_constraint, axis=1)
    print(f"{Fore.GREEN}Found {inconsistencies_flag.sum()} inconsistencies in synthetic data with constraints{Style.RESET_ALL}")
    time.sleep(1)

    # Display rows with violations
    violating_rows = synthetic_data_with_constraints[inconsistencies_flag]

    # Generate polarized data with specific conditions
    print(f"\n{Fore.GREEN}Generating polarized data {Style.RESET_ALL}")

    # Define polarization conditions
    polarization_list = [
        [{"Field": "Sex", "Value": "Female", "Percentage": 25}],
        [{"Field": "Candidate State", "Value": "Hired", "Percentage": 25}],
    ]

    # Generate data with polarization
    final_data = polarized_generation_from_conditions(synthesizer, polarization_list, num_rows=1000)

    # Check the distribution constraints of the polarized data
    check_distribution_constraints(final_data, polarization_list)

    # Define another set of polarization conditions
    polarization_list = [
        [{"Field": "Sex", "Value": "Female", "Percentage": 25},
         {"Field": "Candidate State", "Value": "Hired", "Percentage": 25}],

        [{"Field": "Study Title", "Value": "Five-year degree", "Percentage": 10},
         {"Field": "Assumption Headquarters", "Value": "Milan", "Percentage": 10},
         {"Field": "English", "Value": 3, "Percentage": 10}]
    ]

    # Generate data with the second set of polarization conditions
    final_data = polarized_generation_from_conditions(synthesizer, polarization_list, num_rows=1000)

    # Check the distribution constraints of the new polarized data
    check_distribution_constraints(final_data, polarization_list)
    print()
