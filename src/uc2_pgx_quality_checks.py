from collections import Counter

import pandas as pd


def check_population_representativity_pgx(ground_truth_data, target_column):
    """
    Check population representativity in pharmacogenomics ground truth data.

    Args:
        ground_truth_data: DataFrame containing ground truth phenotype data
        target_column: Column name that indicates population/ethnicity information

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if target_column is None:
        return (
            0,
            "No target column selected. Please select a target column to calculate population representativity.",
        )

    if target_column not in ground_truth_data.columns:
        return 0, f"Target column '{target_column}' not found in ground truth data"

    valid_data = ground_truth_data[~ground_truth_data[target_column].isna()]
    valid_data = valid_data[valid_data[target_column] != "INDETERMINATE"]

    if len(valid_data) == 0:
        return (
            0,
            "No valid data found for target column in ground truth data (excluding INDETERMINATE values)",
        )

    class_counts = valid_data[target_column].value_counts()
    num_classes = len(class_counts)

    if num_classes <= 1:
        return (
            0,
            "Only one population group found in ground truth data. Need at least 2 groups for meaningful representativity.",
        )

    total_samples = len(valid_data)
    ideal_proportion = 1.0 / num_classes

    # Calculate how far each class is from ideal proportion
    proportions = class_counts / total_samples
    deviations = abs(proportions - ideal_proportion)
    max_deviation = deviations.max()

    score = 1.0 - (max_deviation / (1.0 - ideal_proportion))
    score = max(0, min(1, score))

    class_details = []
    for class_name, count in class_counts.items():
        proportion = count / total_samples
        class_details.append(f"'{class_name}': {count} samples ({proportion:.1%})")

    details_str = ", ".join(class_details)
    ideal_percent = ideal_proportion * 100

    return (
        score,
        f"Found {num_classes} population groups in ground truth data. Ideal distribution: {ideal_percent:.1f}% each. Actual distribution: {details_str}. Score: {score:.3f}",
    )


def check_metadata_granularity_pgx(vcf_data, vcf_metadata):
    """
    Check metadata granularity for pharmacogenomics VCF data.

    Args:
        vcf_data: List of DataFrames containing VCF variant data
        vcf_metadata: List of metadata strings for each VCF file

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if vcf_metadata is None:
        return 0, "No VCF metadata provided. Upload metadata to improve this score."

    total_vcf_files = len(vcf_data) if vcf_data else 0
    files_with_metadata = 0

    if vcf_metadata is not None:
        files_with_metadata = sum(
            1 for meta in vcf_metadata if meta and str(meta).strip()
        )

    if total_vcf_files == 0:
        return 0, "No VCF data available"

    ratio = files_with_metadata / total_vcf_files if total_vcf_files > 0 else 0
    return (
        ratio,
        f"VCF files with metadata: {files_with_metadata}/{total_vcf_files} ({ratio:.2%})",
    )


def check_accuracy_pgx(vcf_data, expected_ranges):
    """
    Check accuracy of values in VCF data against expected ranges.

    Args:
        vcf_data: List of DataFrames containing VCF variant data
        expected_ranges: Dictionary mapping column names to (min, max) tuples

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    # If no expected ranges provided, return perfect score
    if not expected_ranges:
        return (
            1.0,
            "No columns selected for accuracy check. Cannot calculate accuracy metric.",
        )

    if not vcf_data or len(vcf_data) == 0:
        return 1.0, "No VCF data provided for accuracy check."

    total_values_checked = 0
    values_within_range = 0
    error_details = []
    columns_checked = 0

    for column, acceptable_values in expected_ranges.items():
        column_found = False
        column_total_values = 0
        column_acceptable_values = 0

        # Check each VCF file for this column
        for vcf_df in vcf_data:
            if column in vcf_df.columns:
                column_found = True
                non_na_values = vcf_df[column].dropna()
                non_na_count = len(non_na_values)
                column_total_values += non_na_count

                acceptable_count = non_na_values.isin(acceptable_values).sum()
                column_acceptable_values += acceptable_count

        if column_found:
            columns_checked += 1
            total_values_checked += column_total_values
            values_within_range += column_acceptable_values

            outside_acceptable = column_total_values - column_acceptable_values
            if outside_acceptable > 0:
                error_details.append(
                    f"{column}: {outside_acceptable}/{column_total_values} values not in acceptable list {acceptable_values}"
                )

    if columns_checked == 0:
        return 1.0, "Selected columns not found in VCF data. Cannot calculate accuracy."

    if total_values_checked == 0:
        return 1.0, "No numeric values found in selected columns across VCF files."

    accuracy_ratio = values_within_range / total_values_checked

    if error_details:
        details_str = ", ".join(error_details)
        return (
            accuracy_ratio,
            f"Checked {columns_checked} column(s) across {len(vcf_data)} VCF files. {values_within_range}/{total_values_checked} values ({accuracy_ratio:.1%}) match acceptable values. Issues: {details_str}",
        )
    else:
        return (
            accuracy_ratio,
            f"Checked {columns_checked} column(s) across {len(vcf_data)} VCF files. All {total_values_checked} values match acceptable values.",
        )


def check_coherence_pgx(vcf_data, column_types):
    """
    Check coherence of data types in VCF data against expected types.

    Args:
        vcf_data: List of DataFrames containing VCF variant data
        column_types: Dictionary mapping column names to expected types ('numeric' or 'categorical')

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not column_types:
        return 1.0, "No column type specifications provided. Cannot check coherence."

    if not vcf_data or len(vcf_data) == 0:
        return 1.0, "No VCF data provided for coherence check."

    total_features = len(column_types)
    consistent_features = 0
    inconsistent_details = []

    for column, expected_type in column_types.items():
        column_found = False
        column_consistent = True
        total_non_numeric = 0
        total_unique_values = 0
        files_with_column = 0

        for vcf_df in vcf_data:
            if column in vcf_df.columns:
                column_found = True
                files_with_column += 1

                if expected_type == "numeric":
                    try:
                        non_na_values = vcf_df[column].dropna()
                        if len(non_na_values) > 0:
                            numeric_conversion = pd.to_numeric(
                                non_na_values, errors="raise"
                            )
                    except Exception:
                        column_consistent = False
                        non_numeric_count = sum(
                            pd.to_numeric(vcf_df[column], errors="coerce").isna()
                            & ~vcf_df[column].isna()
                        )
                        total_non_numeric += non_numeric_count

                elif expected_type == "categorical":
                    # For categorical, check if it has a reasonable number of unique values
                    unique_vals = vcf_df[column].dropna().nunique()
                    total_unique_values = max(total_unique_values, unique_vals)
                    if unique_vals > 50:
                        column_consistent = False

        if column_found:
            if column_consistent:
                consistent_features += 1
            else:
                if expected_type == "numeric" and total_non_numeric > 0:
                    inconsistent_details.append(
                        f"{column}: Expected numeric but contains {total_non_numeric} non-numeric values across {files_with_column} VCF files"
                    )
                elif expected_type == "categorical" and total_unique_values > 50:
                    inconsistent_details.append(
                        f"{column}: Expected categorical but has {total_unique_values} unique values (too many for categorical)"
                    )

    # Calculate coherence as ratio of consistent features
    coherence_ratio = (
        consistent_features / total_features if total_features > 0 else 1.0
    )

    if inconsistent_details:
        details = ", ".join(inconsistent_details)
        return (
            coherence_ratio,
            f"Checked {len(vcf_data)} VCF files. {consistent_features}/{total_features} features have consistent data types. Issues: {details}",
        )
    else:
        return (
            coherence_ratio,
            f"Checked {len(vcf_data)} VCF files. All {total_features} features have consistent data types.",
        )


def check_semantic_coherence_pgx(vcf_data):
    """
    Check semantic coherence of VCF data by detecting duplicate column names.

    Args:
        vcf_data: List of DataFrames containing VCF variant data

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not vcf_data or len(vcf_data) == 0:
        return 1.0, "No VCF data provided for semantic coherence check."

    total_files = len(vcf_data)
    files_with_duplicates = 0
    duplicate_details = []
    total_columns_across_files = 0
    total_unique_columns_across_files = 0

    for i, vcf_df in enumerate(vcf_data):
        total_columns = len(vcf_df.columns)
        total_columns_across_files += total_columns

        # Check for duplicate column names in this VCF file
        duplicate_names = [
            name for name, count in Counter(list(vcf_df.columns)).items() if count > 1
        ]
        num_duplicates = len(duplicate_names)

        if num_duplicates > 0:
            files_with_duplicates += 1
            unique_columns = total_columns - num_duplicates
            duplicate_list = [f"'{name}'" for name in duplicate_names]
            details_str = ", ".join(duplicate_list)
            duplicate_details.append(f"VCF file {i+1}: {details_str}")
            total_unique_columns_across_files += unique_columns
        else:
            total_unique_columns_across_files += total_columns

    if total_columns_across_files == 0:
        return 1.0, "No columns found in VCF data."

    coherence_ratio = total_unique_columns_across_files / total_columns_across_files

    if files_with_duplicates == 0:
        return (
            1.0,
            f"No duplicate column names found across {total_files} VCF files. All column names are unique.",
        )
    else:
        details_str = "; ".join(duplicate_details)
        files_clean = total_files - files_with_duplicates
        return (
            coherence_ratio,
            f"Found duplicate column names in {files_with_duplicates}/{total_files} VCF files. {files_clean} files have all unique column names. Duplicates: {details_str}. Overall coherence: {total_unique_columns_across_files}/{total_columns_across_files} unique columns.",
        )


def check_completeness_pgx(vcf_data):
    """
    Check completeness of VCF data by detecting missing values (including '.' for VCF missing values).

    Args:
        vcf_data: List of DataFrames containing VCF variant data

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not vcf_data or len(vcf_data) == 0:
        return 1.0, "No VCF data provided for completeness check."

    total_values = 0
    missing_values = 0
    missing_by_column = {}

    for i, vcf_df in enumerate(vcf_data):
        file_total_values = vcf_df.size
        total_values += file_total_values

        if file_total_values == 0:
            continue

        # Count standard missing values (NA/NaN) and VCF missing values ('.')
        file_standard_missing = vcf_df.isnull().sum().sum()
        file_vcf_missing = (vcf_df == ".").sum().sum()
        file_total_missing = file_standard_missing + file_vcf_missing
        missing_values += file_total_missing

        for col in vcf_df.columns:
            if col not in missing_by_column:
                missing_by_column[col] = {"missing": 0, "total": 0}

            col_standard_missing = vcf_df[col].isnull().sum()
            col_vcf_missing = (vcf_df[col] == ".").sum()
            col_total_missing = col_standard_missing + col_vcf_missing
            col_total_values = len(vcf_df[col])

            missing_by_column[col]["missing"] += col_total_missing
            missing_by_column[col]["total"] += col_total_values

    if total_values == 0:
        return 1.0, "No data to check."

    complete_values = total_values - missing_values
    completeness_ratio = complete_values / total_values

    missing_columns = {
        col: data for col, data in missing_by_column.items() if data["missing"] > 0
    }

    if len(missing_columns) > 0:
        details = []
        for col, data in missing_columns.items():
            col_percent = (data["missing"] / data["total"]) * 100
            details.append(f"{col}: {data['missing']} ({col_percent:.1f}%)")
        details_str = ", ".join(details)
        return (
            completeness_ratio,
            f"Data completeness across {len(vcf_data)} VCF files: {complete_values}/{total_values} ({completeness_ratio:.1%}). Missing values by column: {details_str}",
        )
    else:
        return (
            completeness_ratio,
            f"No missing values found across {len(vcf_data)} VCF files. Data is 100% complete.",
        )


def check_relational_consistency_pgx(vcf_data):
    """
    Check relational consistency of VCF data by detecting duplicate rows within each VCF file.

    Args:
        vcf_data: List of DataFrames containing VCF variant data

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not vcf_data or len(vcf_data) == 0:
        return 1.0, "No VCF data provided for relational consistency check."

    total_rows_across_files = 0
    total_unique_rows_across_files = 0
    files_with_duplicates = 0
    duplicate_details = []

    for i, vcf_df in enumerate(vcf_data):
        total_rows = len(vcf_df)
        total_rows_across_files += total_rows

        if total_rows == 0:
            continue

        unique_rows = len(vcf_df.drop_duplicates())
        duplicate_rows = total_rows - unique_rows
        total_unique_rows_across_files += unique_rows

        if duplicate_rows > 0:
            files_with_duplicates += 1
            duplicated_mask = vcf_df.duplicated(keep="first")
            duplicated_indices = vcf_df[duplicated_mask].index.tolist()

            # Limit the number of indices shown to avoid clutter
            if len(duplicated_indices) > 10:
                indices_str = str(duplicated_indices[:10])[:-1] + ", ...]"
            else:
                indices_str = str(duplicated_indices)

            duplicate_details.append(
                f"VCF file {i+1}: {duplicate_rows} duplicate rows out of {total_rows} ({(duplicate_rows / total_rows):.1%}), indices: {indices_str}"
            )

    if total_rows_across_files == 0:
        return 1.0, "No data to check."

    consistency_ratio = total_unique_rows_across_files / total_rows_across_files

    if files_with_duplicates > 0:
        details_str = "; ".join(duplicate_details)
        files_clean = len(vcf_data) - files_with_duplicates
        total_duplicates = total_rows_across_files - total_unique_rows_across_files
        return (
            consistency_ratio,
            f"Found duplicate rows in {files_with_duplicates}/{len(vcf_data)} VCF files. {files_clean} files have all unique rows. Total duplicates across all files: {total_duplicates}/{total_rows_across_files} ({(total_duplicates / total_rows_across_files):.1%}). Details: {details_str}",
        )
    else:
        return (
            consistency_ratio,
            f"No duplicate rows found across {len(vcf_data)} VCF files. All {total_rows_across_files} rows are unique within their respective files.",
        )


def run_all_checks_pgx(
    ground_truth_data, vcf_data, vcf_metadata, vcf_filenames, use_case_config
):
    """
    Execute all quantitative quality checks for pharmacogenomics datasets (UC2).

    Args:
        ground_truth_data: DataFrame containing ground truth phenotype data
        vcf_data: List of DataFrames containing VCF variant data
        vcf_metadata: List of metadata strings for each VCF file
        vcf_filenames: List of VCF filenames
        use_case_config: Configuration dictionary for UC2

    Returns:
        Dictionary of check results in format: {metric_name: (score, explanation)}
    """
    results = {}

    results["population_representativity"] = check_population_representativity_pgx(
        ground_truth_data, use_case_config["target_column"]
    )
    results["metadata_granularity"] = check_metadata_granularity_pgx(
        vcf_data, vcf_metadata
    )
    results["accuracy"] = check_accuracy_pgx(
        vcf_data, use_case_config.get("expected_ranges", {})
    )
    results["coherence"] = check_coherence_pgx(
        vcf_data, use_case_config.get("column_types", {})
    )
    results["semantic_coherence"] = check_semantic_coherence_pgx(vcf_data)
    results["completeness"] = check_completeness_pgx(vcf_data)
    results["relational_consistency"] = check_relational_consistency_pgx(vcf_data)

    return results
