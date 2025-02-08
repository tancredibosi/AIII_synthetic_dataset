import pandas as pd
from sdv.constraints import create_custom_constraint_class


def is_valid_YearsHired(column_names, data):
    """
    Check two conditions:
    1. If Candidate State is 'Hired': Year of insertion <= Year of Recruitment
    2. If Candidate State is not 'Hired': Year of Recruitment must be NaN
    
    Args:
        column_names: List containing ['Candidate State', 'Year of insertion', 'Year of Recruitment']
        data: DataFrame containing the columns
        
    Returns:
        pd.Series: Boolean series with True for valid rows
    """
    # Create masks
    hired_mask = data[column_names[0]] == 'Hired'
    not_hired_mask = ~hired_mask
    
    # Initialize result series with True
    validity = pd.Series(True, index=data.index, dtype=bool)
    
    # Condition 1: For hired candidates
    validity.loc[hired_mask] = (
        data[column_names[1]][hired_mask] <= data[column_names[2]][hired_mask]
    )
    
    # Condition 2: For non-hired candidates
    validity.loc[not_hired_mask] = (
        data[column_names[2]][not_hired_mask].isna().astype(bool)
    )
    
    return validity

CustomYearsHired = create_custom_constraint_class(
    is_valid_fn=is_valid_YearsHired,
    # transform_fn=transform, # optional
    # reverse_transform_fn=reverse_transform # optional
)

def is_valid_AgeExperience(column_names, data):
    """
    Check if experience level is valid for the given age range:
    1. If age is '< 20 years': experience cannot be '[+10]', '[7-10]', or '[5-7]'
    2. If age is '20 - 25 years': experience cannot be '[+10]'
    
    Args:
        column_names: List containing ['Age', 'Years Experience']
        data: DataFrame containing the columns
        
    Returns:
        pd.Series: Boolean series with True for valid rows
    """
    # Initialize result series with True
    validity = pd.Series(True, index=data.index)
    
    # Create masks for age ranges
    under_20_mask = data[column_names[0]] == '< 20 years'
    age_20_25_mask = data[column_names[0]] == '20 - 25 years'
    
    # Create masks for invalid experience levels
    high_exp_mask = data[column_names[1]].isin(['[+10]', '[7-10]', '[5-7]'])
    very_high_exp_mask = data[column_names[1]] == '[+10]'
    
    # Set False for invalid combinations
    validity[under_20_mask & high_exp_mask] = False
    validity[age_20_25_mask & very_high_exp_mask] = False
    
    return validity

CustomAgeExperience = create_custom_constraint_class(
    is_valid_fn=is_valid_AgeExperience
)