import pandas as pd
import time
from sdv.sampling import Condition
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from colorama import Fore, Style

from data_plots_utils import plot_comparison_subplots


def exclude_matching_rows(df, exclusion_conditions):
    """
    Removes rows from the DataFrame that match any set of conditions in exclusion_conditions.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        exclusion_conditions (list of lists): Each inner list contains dictionaries with 'Field' and 'Value' keys,
                                             specifying conditions to exclude.

    Returns:
        pd.DataFrame: Filtered DataFrame with matching rows removed.
    """
    # Create an initial mask marking all rows as not removed
    mask_to_remove = pd.Series(False, index=df.index)

    for condition_group in exclusion_conditions:
        # Start with all True mask (assume all rows match initially)
        condition_mask = pd.Series(True, index=df.index)

        for condition in condition_group:
            field, value = condition["Field"], condition["Value"]
            condition_mask &= (df[field] == value)

        # Accumulate rows to remove using OR operation
        mask_to_remove |= condition_mask

    # Return DataFrame with unwanted rows removed
    return df[~mask_to_remove]


def filter_dataframe_by_constraints(df, constraints_list, total_rows):
    """
    Filters the DataFrame based on given constraints, ensuring the required number of rows per constraint.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        constraints_list (list of lists): Each inner list contains dictionaries with 'Field', 'Value', and 'Percentage' keys.
        total_rows (int): Total number of rows to extract based on percentages.

    Returns:
        pd.DataFrame: Filtered DataFrame meeting all constraints.
    """
    final_df = pd.DataFrame()  # Store the filtered results
    used_indices = set()  # Track already used rows to avoid duplication

    for i, constraint_group in enumerate(constraints_list):
        temp_df = df.copy()

        # Exclude rows matching constraints from other groups
        for j, other_group in enumerate(constraints_list):
            if i != j:  # Skip current group
                for constraint in other_group:
                    field, value = constraint["Field"], constraint["Value"]
                    temp_df = temp_df[temp_df[field] != value]

        # Apply current group constraints
        for constraint in constraint_group:
            field, value = constraint["Field"], constraint["Value"]
            num_required = (total_rows * constraint['Percentage']) // 100
            temp_df = temp_df[temp_df[field] == value]

        # Remove already used rows
        temp_df = temp_df.loc[~temp_df.index.isin(used_indices)]

        # Ensure the required number of elements are available
        if len(temp_df) >= num_required:
            temp_df = temp_df.head(num_required)
        else:
            raise ValueError(
                f"Not enough rows for constraints {constraint_group}, missing {num_required - len(temp_df)} elements")

        # Update used indices
        used_indices.update(temp_df.index)

        # Append to final DataFrame
        final_df = pd.concat([final_df, temp_df])

    return final_df.reset_index(drop=True)


def polarized_generation_from_conditions(synthesizer, polarization_list, num_rows=1000, scaling_factor=2, max_retries=3):
    synthetic_data_list = []

    retries = 0
    while retries < max_retries:
        try:
            tot_rows = 0
            for sublist in polarization_list:
                n_elem = (num_rows * sublist[0]['Percentage']) // 100
                tot_rows += n_elem
                prev_percentage = None
                col_values = {}
                for el in sublist:
                    if prev_percentage and prev_percentage != el['Percentage']:
                        raise ValueError("Error: Mismatched percentages")
                    col_values[el['Field']] = el['Value']

                condition = Condition(
                    num_rows=n_elem * scaling_factor,
                    column_values=col_values
                )
                polarized_synthetic_data = synthesizer.sample_from_conditions(
                    conditions=[condition],
                )
                synthetic_data_list.append(polarized_synthetic_data)

            # Combine all generated synthetic data
            all_polarized_data = pd.concat(synthetic_data_list).reset_index(drop=True)
            filtered_polarized_data = filter_dataframe_by_constraints(all_polarized_data, polarization_list, num_rows)
            break
        except Exception as e:
            retries += 1
            scaling_factor *= 2  # Increase scaling factor on retry
            print(f"{Fore.YELLOW}Retry {retries}/{max_retries}: Increasing scaling factor to {scaling_factor} due to error: {e}{Style.RESET_ALL}")
            time.sleep(1)
            if retries == max_retries:
                raise RuntimeError("Max retries reached. Unable to generate valid polarized synthetic data.")

    scaling_factor = 3
    retries = 0
    while retries < max_retries:
        try:
            synthetic_data = synthesizer.sample(num_rows=num_rows * scaling_factor)
            filtered_synthetic_data = exclude_matching_rows(synthetic_data, polarization_list)
            fill_values = filtered_synthetic_data.sample(n=num_rows-tot_rows)
            break
        except Exception as e:
            retries += 1
            scaling_factor *= 2  # Increase scaling factor on retry
            print(f"{Fore.YELLOW}Retry {retries}/{max_retries}: Increasing scaling factor to {scaling_factor} due to error: {e}{Style.RESET_ALL}")
            time.sleep(1)
            if retries == max_retries:
                raise RuntimeError("Max retries reached. Unable to generate valid polarized synthetic data.")

    final_data = pd.concat([filtered_polarized_data, fill_values]).reset_index(drop=True)
    return final_data


def polarized_generation_from_columns(synthesizer, polarization_list, num_rows=1000):
    synthetic_data = synthesizer.sample(num_rows=num_rows*5)

    reference_data = pd.DataFrame()
    for sublist in polarization_list:
        prev_percentage = None
        exclude_conditions = {}
        new_rows = pd.DataFrame()
        for el in sublist:
            if prev_percentage and prev_percentage != el['Percentage']:
                raise ValueError("Error: Mismatched percentages")

            exclude_conditions[el['Field']] = el['Value']
            n_elem = (num_rows * el['Percentage']) // 100
            polarized_rows = pd.DataFrame({el['Field']: [el['Value']] * n_elem})
            new_rows = pd.concat([new_rows, polarized_rows], axis=1).reset_index(drop=True)
            prev_percentage = el['Percentage']

        mask = ~synthetic_data.apply(lambda row: all(row[col] == val for col, val in exclude_conditions.items()), axis=1)
        fill_values = synthetic_data[mask].sample(n=num_rows-n_elem)
        new_rows = pd.concat([new_rows, fill_values[new_rows.columns]]).reset_index(drop=True)
        reference_data = pd.concat([reference_data, new_rows], axis=1).reset_index(drop=True)
        reference_data = reference_data.sample(frac=1).reset_index(drop=True)

    polarized_synthetic_data = synthesizer.sample_remaining_columns(
        known_columns=reference_data,
        max_tries_per_batch=500,
        batch_size=1024,
    )
    return polarized_synthetic_data


def compare_synthesizer(s1, s2, metadata, data, num_rows=1000):
    s1.fit(data)
    s2.fit(data)
    
    synthetic_data_s1 = s1.sample(num_rows=num_rows)
    synthetic_data_s2 = s2.sample(num_rows=num_rows)
    
    diagnostic_report_s1 = run_diagnostic(data, synthetic_data_s1, metadata, verbose=False)
    drs1_dict = dict(zip(diagnostic_report_s1.get_properties()['Property'], diagnostic_report_s1.get_properties()['Score']))
    
    quality_report_s1 = evaluate_quality(data, synthetic_data_s1, metadata, verbose=False)
    dqs1_dict = dict(zip(quality_report_s1.get_properties()['Property'], quality_report_s1.get_properties()['Score']))
    
    diagnostic_report_s2 = run_diagnostic(data, synthetic_data_s2, metadata, verbose=False)
    drs2_dict = dict(zip(diagnostic_report_s2.get_properties()['Property'], diagnostic_report_s2.get_properties()['Score']))
    
    quality_report_s2 = evaluate_quality(data, synthetic_data_s2, metadata, verbose=False)
    dqs2_dict = dict(zip(quality_report_s2.get_properties()['Property'], quality_report_s2.get_properties()['Score']))
    
    plot_comparison_subplots(drs1_dict, drs2_dict, dqs1_dict, dqs2_dict, 
                             dict1_name=f"{s1.__class__.__name__}",
                             dict2_name=f"{s2.__class__.__name__}")
