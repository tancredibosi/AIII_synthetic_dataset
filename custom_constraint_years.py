import pandas as pd
from sdv.constraints import create_custom_constraint_class


def is_valid(column_names, data):
    """
    Check if Year of insertion <= Year of Recruitment when Candidate State is 'Hired'
    
    Args:
        column_names: List containing ['Candidate State', 'Year of insertion', 'Year of Recruitment']
        data: DataFrame containing the columns
        
    Returns:
        pd.Series: Boolean series with True for valid rows
    """
    # Create a mask for rows where Candidate State is 'Hired'
    hired_mask = data[column_names[0]] == 'Hired'
    
    # Initialize result series with True
    validity = pd.Series(True, index=data.index)
    
    # For hired candidates, check if Year of insertion <= Year of Recruitment
    validity[hired_mask] = data[column_names[1]][hired_mask] <= data[column_names[2]][hired_mask]
    
    return validity

CustomYearsHired = create_custom_constraint_class(
    is_valid_fn=is_valid,
    # transform_fn=transform, # optional
    # reverse_transform_fn=reverse_transform # optional
)