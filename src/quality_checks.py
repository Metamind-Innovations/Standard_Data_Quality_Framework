import pandas as pd
import numpy as np
from collections import Counter

def check_population_representativity(data, target_column):
    if target_column not in data.columns:
        return 0, "Target column not found in data"
    
    # Remove NA values
    valid_data = data[~data[target_column].isna()]
    
    if len(valid_data) == 0:
        return 0, "No valid data found for target column"
    
    class_counts = valid_data[target_column].value_counts()
    if len(class_counts) <= 1:
        return 0, "Only one class found in data"
    
    minority_samples = class_counts.min()
    majority_samples = class_counts.max()
    ratio = minority_samples / majority_samples
    
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    return ratio, f"Minority class '{minority_class}': {minority_samples} samples, Majority class '{majority_class}': {majority_samples} samples. Ratio: {ratio:.3f}"

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
    total_samples = len(data)
    erroneous_samples_count = 0
    error_details = []
    
    for column, value_range in expected_ranges.items():
        if column in data.columns:
            min_val, max_val = value_range
            # Get only numeric values that are outside the expected range
            numeric_data = pd.to_numeric(data[column], errors='coerce')
            invalid_mask = (numeric_data < min_val) | (numeric_data > max_val)
            invalid_values = data[invalid_mask].dropna()
            erroneous_count = len(invalid_values)
            erroneous_samples_count += erroneous_count
            
            if erroneous_count > 0:
                error_details.append(f"{column}: {erroneous_count} values outside range [{min_val}, {max_val}]")
    
    ratio = erroneous_samples_count / (total_samples * len(expected_ranges)) if total_samples > 0 and len(expected_ranges) > 0 else 0
    
    if error_details:
        details_str = ", ".join(error_details)
        return 1 - ratio, f"Found {erroneous_samples_count} values outside expected ranges: {details_str}"
    else:
        return 1.0, "All values within expected ranges"

def check_coherence(data, column_types):
    total_features = len(column_types)
    inconsistent_features = 0
    inconsistent_details = []
    
    for column, expected_type in column_types.items():
        if column in data.columns:
            if expected_type == "numeric":
                try:
                    # Check if all non-NA values can be converted to numeric
                    non_na_values = data[column].dropna()
                    numeric_conversion = pd.to_numeric(non_na_values, errors='raise')
                except Exception:
                    inconsistent_features += 1
                    non_numeric_count = sum(pd.to_numeric(data[column], errors='coerce').isna() & ~data[column].isna())
                    inconsistent_details.append(f"{column}: Expected numeric but contains {non_numeric_count} non-numeric values")
            elif expected_type == "categorical":
                # For categorical, we can check if it has a limited number of unique values
                unique_vals = data[column].dropna().nunique()
                if unique_vals > 10:  # Arbitrary threshold for categorical
                    inconsistent_features += 1
                    inconsistent_details.append(f"{column}: Expected categorical but has {unique_vals} unique values")
    
    ratio = inconsistent_features / total_features if total_features > 0 else 0
    
    if inconsistent_details:
        details = ", ".join(inconsistent_details)
        return 1 - ratio, f"Found {inconsistent_features} inconsistent features: {details}"
    else:
        return 1.0, "All features have consistent data types"

def check_semantic_coherence_option1(data):
    total_columns = len(data.columns)
    detected_columns = 0
    detected_details = []
    
    # Check for columns with different names but identical values
    column_pairs = []
    for i in range(total_columns):
        for j in range(i+1, total_columns):
            col1, col2 = data.columns[i], data.columns[j]
            
            # Skip if either column has all NaN values
            if data[col1].isna().all() or data[col2].isna().all():
                continue
                
            # Check if non-NA values are equal
            non_na_mask = ~(data[col1].isna() | data[col2].isna())
            if non_na_mask.sum() > 0:
                if data.loc[non_na_mask, col1].equals(data.loc[non_na_mask, col2]):
                    detected_columns += 2
                    column_pairs.append((col1, col2))
                    detected_details.append(f"{col1} and {col2} have identical values")
    
    ratio = detected_columns / total_columns if total_columns > 0 else 0
    
    if detected_details:
        return ratio, f"Found {len(detected_details)} column pairs with different names but same values: {', '.join(detected_details)}"
    else:
        return ratio, "No columns with different names but same values found"

def check_semantic_coherence_option2(data):
    # This function checks for duplicate column names
    total_columns = len(data.columns)
    duplicate_names = [name for name, count in Counter(list(data.columns)).items() if count > 1]
    detected_columns = len(duplicate_names)
    
    if duplicate_names:
        detected_details = [f"Duplicate column name: {name}" for name in duplicate_names]
        details_str = ", ".join(detected_details)
        return detected_columns / total_columns, f"Found {detected_columns} duplicate column names: {details_str}"
    else:
        return 0, "No duplicate column names found"

def check_completeness(data):
    total_values = data.size
    missing_values = data.isnull().sum().sum()
    
    # Get missing values per column for more detailed report
    missing_by_column = data.isnull().sum()
    missing_columns = missing_by_column[missing_by_column > 0]
    
    ratio = missing_values / total_values if total_values > 0 else 0
    
    if len(missing_columns) > 0:
        details = ", ".join([f"{col}: {count} missing" for col, count in missing_columns.items()])
        return 1 - ratio, f"Missing values: {missing_values}/{total_values} ({ratio:.2%}). Details: {details}"
    else:
        return 1.0, "No missing values found"

def check_relational_consistency(data):
    total_rows = len(data)
    unique_rows = len(data.drop_duplicates())
    duplicate_rows = total_rows - unique_rows
    
    if duplicate_rows > 0:
        # Find which rows are duplicated
        duplicated_mask = data.duplicated(keep='first')
        duplicated_indices = data[duplicated_mask].index.tolist()
        
        ratio = duplicate_rows / total_rows if total_rows > 0 else 0
        return ratio, f"Found {duplicate_rows} duplicate rows out of {total_rows} ({ratio:.2%}). Duplicate indices: {duplicated_indices}"
    else:
        return 0, f"No duplicate rows found"

def run_all_checks(data, use_case_config, metadata=None):
    results = {}
    
    target_column = use_case_config.get("target_column", None)
    expected_ranges = use_case_config.get("expected_ranges", {})
    column_types = use_case_config.get("column_types", {})
    
    results["population_representativity"] = check_population_representativity(data, target_column)
    results["metadata_granularity"] = check_metadata_granularity(data, metadata)
    results["accuracy"] = check_accuracy(data, expected_ranges)
    results["coherence"] = check_coherence(data, column_types)
    results["semantic_coherence_option1"] = check_semantic_coherence_option1(data)
    results["semantic_coherence_option2"] = check_semantic_coherence_option2(data)
    results["completeness"] = check_completeness(data)
    results["relational_consistency"] = check_relational_consistency(data)
    
    return results