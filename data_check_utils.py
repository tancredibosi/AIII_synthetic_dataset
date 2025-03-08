import re
import time
import pandas as pd
from colorama import Fore, Style


def extract_number(value, choose_second=False):
    """
    Extracts the first or second number from a string.

    Args:
        value (str): The string containing numbers.
        choose_second (bool): Flag to decide whether to return the second number (if any).

    Returns:
        int or None: The extracted number or None if no numbers are found.
    """
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


def check_constraint(row):
    """
    Checks if a row violates any constraints.

    Args:
        row (pd.Series): A row of data to check for constraint violations.

    Returns:
        bool: True if any constraint violation is found, False otherwise.
    """
    flag = False
    flag += minor_worker_check(row)  # Check for minor worker constraint
    flag += hired_check(row)  # Check for hired status constraint
    return bool(flag)  # Return True if any constraint is violated


def minor_worker_check(row):
    """
    Checks if the worker was a minor when starting the job.

    Args:
        row (pd.Series): A row of data containing age and work experience.

    Returns:
        bool: True if the worker was a minor at the time of hire, False otherwise.
    """
    age = extract_number(row['Age Range'], choose_second=True)
    work_exp = extract_number(row['Years Experience'])
    if work_exp == 'nan':
        return False
    # The worker is a minor if their age at the start of work is less than 18
    return (age - work_exp) < 18


def hired_check(row):
    """
    Checks if the 'Years of Recruitment' is less than or equal to 'Years of Insertion'
    when the candidate's state is 'Hired'. If not hired, returns True for constraint violation.

    Args:
        row (pd.Series): A row of data with candidate state and year information.

    Returns:
        bool: True if the constraint is violated, False otherwise.
    """
    candidate_state = row['Candidate State']
    years_insert = row['Year of insertion']
    years_recruit = row['Year of Recruitment']

    if candidate_state == 'Hired':
        if pd.notna(years_insert) and pd.notna(years_recruit):
            if years_insert > years_recruit:
                return True  # Violation: Years of Insertion is greater than Recruitment
    else:
        if pd.notna(years_recruit):
            return True  # Violation: 'Years of Recruitment' should be NaN if not hired
    return False


def check_distribution_constraints(df, constraints_list):
    """
    Checks if the distribution constraints are respected within the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to check.
        constraints_list (list): A list of constraints (field-value pairs) to validate.

    Returns:
        bool: True if all distribution constraints are satisfied, False if any are violated.
    """
    total_rows = len(df)
    if total_rows == 0:
        return False  # Return False if the dataframe is empty

    for constraint_group in constraints_list:
        filtered_df = df.copy()

        # Filter the dataframe based on each condition in the constraint group
        for condition in constraint_group:
            field, value = condition["Field"], condition["Value"]
            filtered_df = filtered_df[filtered_df[field] == value]

        actual_count = len(filtered_df)
        actual_percentage = (actual_count / total_rows) * 100
        expected_percentage = constraint_group[0]["Percentage"]
        expected_count = round((expected_percentage / 100) * total_rows)

        # Compare the actual distribution with the expected distribution
        if round(actual_percentage, 2) != round(expected_percentage, 2):
            print(f"{Fore.RED}Distribution of polarization not respected{Style.RESET_ALL}")
            print(f"Condition: {constraint_group}")
            print(f"Actual Count: {actual_count}, Expected Count: {expected_count}")
            print(f"Actual Percentage: {actual_percentage}%, Expected Percentage: {expected_percentage}%")
            return False  # Constraint is not met

    print(f"{Fore.GREEN}Distribution of polarization respected{Style.RESET_ALL}")
    time.sleep(1)  # Add delay for better visualization of the result
    return True  # All constraints are satisfied
