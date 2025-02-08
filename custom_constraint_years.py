import pandas as pd
from sdv.constraints import create_custom_constraint_class


def is_valid(column_names, data):
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
    validity = pd.Series(True, index=data.index)
    
    # Condition 1: For hired candidates, check if Year of insertion <= Year of Recruitment
    validity[hired_mask] = data[column_names[1]][hired_mask] <= data[column_names[2]][hired_mask]
    
    # Condition 2: For non-hired candidates, check if Year of Recruitment is NaN
    validity[not_hired_mask] = data[column_names[2]][not_hired_mask].isna()
    
    return validity

CustomYearsHired = create_custom_constraint_class(
    is_valid_fn=is_valid,
    # transform_fn=transform, # optional
    # reverse_transform_fn=reverse_transform # optional
)