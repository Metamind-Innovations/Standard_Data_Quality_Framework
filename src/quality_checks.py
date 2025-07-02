from collections import Counter

import pandas as pd


def check_population_representativity(data, target_column):
    if target_column is None:
        return 0, "No target column selected. Please select a target column to calculate population representativity."

    if target_column not in data.columns:
        return 0, "Target column not found in data"

    # Remove NA values
    valid_data = data[~data[target_column].isna()]

    if len(valid_data) == 0:
        return 0, "No valid data found for target column"

    class_counts = valid_data[target_column].value_counts()
    num_classes = len(class_counts)

    if num_classes <= 1:
        return 0, "Only one class found in data. Need at least 2 classes for meaningful representativity."

    total_samples = len(valid_data)
    ideal_proportion = 1.0 / num_classes

    # Calculate how far each class is from ideal proportion
    proportions = class_counts / total_samples
    deviations = abs(proportions - ideal_proportion)
    max_deviation = deviations.max()

    # Score based on maximum deviation from ideal
    # Perfect balance (0 deviation) = 1.0
    # Maximum imbalance (one class has everything) = 0.0
    score = 1.0 - (max_deviation / (1.0 - ideal_proportion))
    score = max(0, min(1, score))  # Ensure between 0 and 1

    # Create detailed explanation
    class_details = []
    for class_name, count in class_counts.items():
        proportion = count / total_samples
        class_details.append(f"'{class_name}': {count} samples ({proportion:.1%})")

    details_str = ", ".join(class_details)
    ideal_percent = ideal_proportion * 100

    return score, f"Found {num_classes} classes. Ideal distribution: {ideal_percent:.1f}% each. Actual distribution: {details_str}. Score: {score:.3f}"


def check_metadata_granularity(data, metadata=None):
    if metadata is None:
        return 0, "No metadata provided. Upload metadata to improve this score."

    total_patients = len(data)
    patients_with_metadata = len(metadata) if metadata is not None else 0

    if total_patients == 0:
        return 0, "No patient data available"

    ratio = patients_with_metadata / total_patients if total_patients > 0 else 0
    return ratio, f"Patients with metadata: {patients_with_metadata}/{total_patients} ({ratio:.2%})"


def check_accuracy(data, expected_ranges):
    # If no expected ranges provided (no age or height columns selected), return perfect score
    if not expected_ranges:
        return 1.0, "No columns selected for accuracy check. Cannot calculate accuracy metric."

    total_samples = len(data)
    total_values_checked = 0
    values_within_range = 0
    error_details = []
    columns_checked = 0

    for column, value_range in expected_ranges.items():
        if column in data.columns:
            columns_checked += 1
            min_val, max_val = value_range
            # Get only numeric values
            numeric_data = pd.to_numeric(data[column], errors='coerce')
            # Count non-NA values for this column
            non_na_values = numeric_data.dropna()
            non_na_count = len(non_na_values)
            total_values_checked += non_na_count

            # Count values within range
            within_range = ((non_na_values >= min_val) & (non_na_values <= max_val)).sum()
            values_within_range += within_range

            outside_range = non_na_count - within_range
            if outside_range > 0:
                error_details.append(
                    f"{column}: {outside_range}/{non_na_count} values outside range [{min_val}, {max_val}]")

    # Calculate ratio based on columns that were actually checked
    if columns_checked == 0:
        return 1.0, "Selected columns not found in data. Cannot calculate accuracy."

    if total_values_checked == 0:
        return 1.0, "No numeric values found in selected columns."
    
    # Calculate accuracy as proportion of values within range
    accuracy_ratio = values_within_range / total_values_checked

    if error_details:
        details_str = ", ".join(error_details)
        return accuracy_ratio, f"Checked {columns_checked} column(s). {values_within_range}/{total_values_checked} values ({accuracy_ratio:.1%}) within expected ranges. Issues: {details_str}"
    else:
        return accuracy_ratio, f"Checked {columns_checked} column(s). All {total_values_checked} values within expected ranges."


def check_coherence(data, column_types):
    if not column_types:
        return 1.0, "No column type specifications provided. Cannot check coherence."

    total_features = len(column_types)
    consistent_features = 0
    inconsistent_details = []

    for column, expected_type in column_types.items():
        if column in data.columns:
            if expected_type == "numeric":
                try:
                    # Check if all non-NA values can be converted to numeric
                    non_na_values = data[column].dropna()
                    if len(non_na_values) > 0:
                        numeric_conversion = pd.to_numeric(non_na_values, errors='raise')
                        consistent_features += 1
                except Exception:
                    non_numeric_count = sum(pd.to_numeric(data[column], errors='coerce').isna() & ~data[column].isna())
                    inconsistent_details.append(
                        f"{column}: Expected numeric but contains {non_numeric_count} non-numeric values")
            elif expected_type == "categorical":
                # For categorical, we can check if it has a limited number of unique values
                unique_vals = data[column].dropna().nunique()
                if unique_vals <= 50:
                    consistent_features += 1
                else:
                    inconsistent_details.append(f"{column}: Expected categorical but has {unique_vals} unique values")
    
    # Calculate coherence as ratio of consistent features
    coherence_ratio = consistent_features / total_features if total_features > 0 else 1.0

    if inconsistent_details:
        details = ", ".join(inconsistent_details)
        return coherence_ratio, f"{consistent_features}/{total_features} features have consistent data types. Issues: {details}"
    else:
        return coherence_ratio, f"All {total_features} features have consistent data types."


def check_semantic_coherence(data):
    # This function checks for duplicate column names
    total_columns = len(data.columns)
    duplicate_names = [name for name, count in Counter(list(data.columns)).items() if count > 1]
    num_duplicates = len(duplicate_names)

    # Calculate score: 1.0 if no duplicates, lower if duplicates exist
    if num_duplicates == 0:
        return 1.0, "No duplicate column names found. All column names are unique."
    else:
        # Calculate ratio of unique columns
        unique_columns = total_columns - num_duplicates
        ratio = unique_columns / total_columns

        duplicate_details = [f"'{name}'" for name in duplicate_names]
        details_str = ", ".join(duplicate_details)
        return ratio, f"Found {num_duplicates} duplicate column names: {details_str}. {unique_columns}/{total_columns} columns have unique names."


def check_completeness(data):
    total_values = data.size
    if total_values == 0:
        return 1.0, "No data to check."

    missing_values = data.isnull().sum().sum()
    complete_values = total_values - missing_values
    
    # Calculate completeness ratio
    completeness_ratio = complete_values / total_values
    
    # Get missing values per column for more detailed report
    missing_by_column = data.isnull().sum()
    missing_columns = missing_by_column[missing_by_column > 0]

    if len(missing_columns) > 0:
        details = []
        for col, count in missing_columns.items():
            col_total = len(data)
            col_percent = (count / col_total) * 100
            details.append(f"{col}: {count} ({col_percent:.1f}%)")
        details_str = ", ".join(details)
        return completeness_ratio, f"Data completeness: {complete_values}/{total_values} ({completeness_ratio:.1%}). Missing values by column: {details_str}"
    else:
        return completeness_ratio, "No missing values found. Data is 100% complete."


def check_relational_consistency(data):
    total_rows = len(data)
    if total_rows == 0:
        return 1.0, "No data to check."

    unique_rows = len(data.drop_duplicates())
    duplicate_rows = total_rows - unique_rows
    
    # Calculate consistency ratio (higher is better - no duplicates)
    consistency_ratio = unique_rows / total_rows

    if duplicate_rows > 0:
        # Find which rows are duplicated
        duplicated_mask = data.duplicated(keep='first')
        duplicated_indices = data[duplicated_mask].index.tolist()
        
        # Limit the number of indices shown to avoid clutter
        if len(duplicated_indices) > 10:
            indices_str = str(duplicated_indices[:10])[:-1] + ", ...]"
        else:
            indices_str = str(duplicated_indices)

        return consistency_ratio, f"Found {duplicate_rows} duplicate rows out of {total_rows} ({(duplicate_rows / total_rows):.1%}). Unique rows: {unique_rows}/{total_rows}. Duplicate indices: {indices_str}"
    else:
        return consistency_ratio, f"No duplicate rows found. All {total_rows} rows are unique."


def check_clinical_stage_consistency(data):
    """
    UC1-specific check for clinical staging consistency.

    :param data: DataFrame containing clinical data
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    stage_columns = ['clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 'Overall.Stage']
    available_columns = [col for col in stage_columns if col in data.columns]

    if len(available_columns) < 2:
        return 1.0, "Insufficient staging columns for consistency check"

    inconsistencies = 0
    total_checked = 0

    for idx, row in data.iterrows():
        if all(pd.notna(row.get(col)) for col in available_columns):
            total_checked += 1
            t_stage = row.get('clinical.T.Stage')
            n_stage = row.get('Clinical.N.Stage')
            m_stage = row.get('Clinical.M.Stage')
            overall = row.get('Overall.Stage')

            if pd.notna(t_stage) and pd.notna(n_stage) and pd.notna(m_stage) and pd.notna(overall):
                expected_overall = determine_overall_stage(t_stage, n_stage, m_stage)
                if expected_overall and str(overall) != str(expected_overall):
                    inconsistencies += 1

    if total_checked == 0:
        return 1.0, "No complete staging data found for consistency check"

    consistency_ratio = (total_checked - inconsistencies) / total_checked

    return consistency_ratio, f"Clinical staging consistency: {total_checked - inconsistencies}/{total_checked} ({consistency_ratio:.1%}) cases have consistent staging. {inconsistencies} inconsistencies found."


def determine_overall_stage(t_stage, n_stage, m_stage):
    """
    Simplified NSCLC staging logic for consistency checking.

    :param t_stage: T stage value
    :param n_stage: N stage value
    :param m_stage: M stage value
    :return: Expected overall stage
    :rtype: str or None
    """
    try:
        t = int(t_stage) if pd.notna(t_stage) else 0
        n = int(n_stage) if pd.notna(n_stage) else 0
        m = int(m_stage) if pd.notna(m_stage) else 0

        if m > 0:
            return "IV"
        elif t >= 4 or n >= 3:
            return "IIIb"
        elif n == 2 or (t == 3 and n <= 1):
            return "IIIa"
        elif t >= 2 and n <= 1:
            return "II"
        elif t == 1 and n == 0:
            return "I"
        else:
            return None
    except:
        return None


def run_all_checks(data, use_case_config, metadata=None):
    results = {}

    target_column = use_case_config.get("target_column", None)
    expected_ranges = use_case_config.get("expected_ranges", {})
    column_types = use_case_config.get("column_types", {})

    results["population_representativity"] = check_population_representativity(data, target_column)
    results["metadata_granularity"] = check_metadata_granularity(data, metadata)
    results["accuracy"] = check_accuracy(data, expected_ranges)
    results["coherence"] = check_coherence(data, column_types)
    results["semantic_coherence"] = check_semantic_coherence(data)
    results["completeness"] = check_completeness(data)
    results["relational_consistency"] = check_relational_consistency(data)

    if use_case_config.get("name") == "DuneAI":
        results["clinical_stage_consistency"] = check_clinical_stage_consistency(data)

    return results
