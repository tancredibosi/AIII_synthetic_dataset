import re
import time
import pandas as pd
from colorama import Fore, Style


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
    time.sleep(1)
    return True  # All constraints are satisfied