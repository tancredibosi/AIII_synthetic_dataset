import pandas as pd
from sdv.constraints import create_custom_constraint_class # type: ignore


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