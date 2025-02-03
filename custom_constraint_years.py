import pandas as pd
from sdv.constraints import create_custom_constraint_class


def is_valid(column_names, data):

    if data[column_names[0]] == 'Hired':
        validity = data[column_names[1]] <= data[column_names[2]]
    else:
        validity = data[column_names[2]].isna()

    return pd.Series(validity)

MyCustomConstraintClass = create_custom_constraint_class(
    is_valid_fn=is_valid,
    # transform_fn=transform, # optional
    # reverse_transform_fn=reverse_transform # optional
)