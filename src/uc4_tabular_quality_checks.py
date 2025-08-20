import pandas as pd
from typing import Tuple, Dict, Any, Optional
from collections import Counter


def _calculate_representativity_score(
    cat_counts: pd.DataFrame,
) -> Tuple[float, pd.DataFrame, float]:
    """
    Calculate a balance score indicating how evenly samples are distributed among categories.

    This function measures representativity by comparing the actual category proportions
    to an ideal balanced distribution. A perfect balance (equal counts across categories)
    returns a score of 1.0, while maximum imbalance (all samples in one category) returns 0.0.

    Args:
        cat_counts (pd.DataFrame): DataFrame containing at least a 'count' column
            with the sample counts per category.

    Returns:
        tuple[float, pd.DataFrame]:
            - Representativity score between 0.0 and 1.0
            - Updated DataFrame with 'act_prop' and 'abs_dev' columns included.
            - Ideal proportion if all categories were perfectly balanced.
    """

    num_categories = len(cat_counts)  # Number of unique categories
    tot_samples = sum(
        cat_counts["count"]
    )  # Total number of samples across all categories

    # Ideal proportion if all categories were perfectly balanced
    ideal_proportion = 1 / num_categories

    # Calculate actual proportion for each category
    cat_counts["act_prop"] = cat_counts["count"] / tot_samples

    # Calculate absolute deviation from the ideal proportion for each category
    cat_counts["abs_dev"] = (ideal_proportion - cat_counts["act_prop"]).abs()

    # Total deviation across all categories
    total_deviation = sum(cat_counts["abs_dev"])

    # Maximum possible deviation (case where one category has all samples, others have zero)
    max_deviation = 2 * (1 - ideal_proportion)

    # Convert deviation into balance score: 1 = perfect balance, 0 = maximum imbalance
    balance_score = 1.0 - (total_deviation / max_deviation)

    # Ensure the score stays within [0.0, 1.0]
    return max(0.0, min(1.0, balance_score)), cat_counts, ideal_proportion


def _analyze_categorical_representativity(
    data: pd.DataFrame, column: str, feature_name: str
) -> Tuple[float, str, dict]:
    """
    Analyze representativity for categorical features.

    This function evaluates how balanced the distribution of categories is for a given feature
    by comparing actual category proportions to an ideal balanced distribution. It returns
    a representativity score, an explanatory string, and detailed statistics.

    Args:
        data (pd.DataFrame):
            DataFrame containing the categorical data.
        column (str):
            Name of the column to analyze.
        feature_name (str):
            Human-readable feature name for use in explanations.

    Returns:
        tuple[float, str, dict]:
            - float: Representativity score between 0.0 (maximum imbalance) and 1.0 (perfect balance).
            - str: Explanation of the score and category distribution.
            - dict: Detailed results dictionary.
    """

    # Remove NaN values to ensure only valid data is analyzed
    col_data = data[column].copy().dropna()
    total_samples = len(col_data)

    # If there are no valid samples, return a score of 0
    if total_samples == 0:
        return 0, f"No valid data for {feature_name}", {}

    # Count the number of samples in each category
    category_counts = col_data.value_counts()
    category_counts_dict = category_counts.to_dict()

    # Convert counts to DataFrame for scoring function
    category_counts = category_counts.reset_index()

    num_categories = len(category_counts)

    # If only one category exists, representativity is zero
    if num_categories <= 1:
        return 0, f"Only one {feature_name.lower()} category found", {}

    # Calculate balance score and ideal proportion
    score, category_counts, ideal_proportion = _calculate_representativity_score(
        category_counts
    )

    # Prepare category details for explanation
    categ_details = []
    for _, row in category_counts.iterrows():
        categ_details.append(f"'{row[column]}': {row["count"]} ({row["act_prop"]:.1%})")

    # Limit explanation to first three categories for readability
    details_str = ", ".join(categ_details[:3])
    if len(categ_details) > 3:
        details_str += f" and {len(categ_details) - 3} more"

    # Create explanation string
    feat_name_to_show = (
        feature_name if feature_name[0].isupper() else feature_name.capitalize()
    )
    explanation = f"{feat_name_to_show} balance score {score:.3f} (ideal: {ideal_proportion:.1%} each), Distribution: {details_str}"

    # Collect detailed results
    details = {
        "balance_score": score,
        "ideal_proportion": ideal_proportion,
        "classes": category_counts_dict,
        "num_classes": num_categories,
    }

    return score, explanation, details


def _analyze_age_representativity(
    data: pd.DataFrame, column: str, feature_name: str
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Analyze the representativity (balance) of an age-related feature by grouping ages into predefined bins.

    Args:
        data (pd.DataFrame): The input DataFrame containing the feature.
        column (str): The name of the age column in the DataFrame.
        feature_name (str): A human-readable name for the feature (used in messages).

    Returns:
        Tuple[float, str, Dict[str, Any]]:
            - Balance score between 0.0 and 1.0
            - Text explanation summarizing the results
            - Dictionary containing detailed results
    """

    # Convert column to numeric, replacing invalid entries with NaN, then drop missing values
    age_data = pd.to_numeric(data[column], errors="coerce").dropna()
    total_samples = len(age_data)

    # If no valid ages remain, return 0 score
    if total_samples == 0:
        return 0, f"No valid data for {feature_name}", {}

    # Define age bins and labels
    age_bins = [0, 40, 55, 70, 120]
    age_labels = ["<40", "40-54", "55-69", "70+"]

    # Group ages into categories
    age_groups = pd.cut(
        age_data, bins=age_bins, labels=age_labels, include_lowest=True, right=True
    )

    # Count patients per age group
    age_counts = age_groups.value_counts()
    age_counts_dict = age_counts.to_dict()

    # Keep only non-empty groups for score calculation
    non_empty_age_counts = age_counts[age_counts > 0]
    num_age_categories = len(non_empty_age_counts)

    # If there is not enough diversity in age groups, return 0 score
    if num_age_categories <= 1:
        return 0, f"Insufficient {feature_name.lower()} group diversity", {}

    # Prepare counts for balance score calculation
    non_empty_age_counts = non_empty_age_counts.reset_index()

    # Calculate balance score
    score, non_empty_age_counts, ideal_proportion = _calculate_representativity_score(
        non_empty_age_counts
    )

    # Prepare summary of distribution for explanation
    age_group_details = []
    for _, row in non_empty_age_counts.iterrows():
        age_group_details.append(
            f"'{row[column]}': {row["count"]} ({row["act_prop"]:.1%})"
        )

    details_str = ", ".join(age_group_details[:3])
    if len(age_group_details) > 3:
        details_str += f" and {len(age_group_details) - 3} more"

    explanation = f"Age balance score {score:.3f} (ideal: {ideal_proportion:.1%} per group), Distribution: {details_str}"

    # Costruct details dictionary
    details = {
        "balance_score": score,
        "ideal_proportion": ideal_proportion,
        "age_groups": age_counts_dict,
        "mean_age": float(age_data.mean()),
        "age_range": [float(age_data.min()), float(age_data.max())],
    }

    return score, explanation, details


def check_population_representativity_tabular(
    tabular_data: pd.DataFrame,
    target_data: pd.DataFrame,
    selected_features: Dict[str, Optional[str]],
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Compute a population representativity score for tabular datasets.
    This function evaluates how balanced the dataset is across selected demographic
    or target features (e.g., target, gender, subpopulation, age).
    Each feature is scored independently, and the final score is the average of all
    valid feature scores.

    Args:
        tabular_data (pd.DataFrame): DataFrame containing clinical data.
        target_data (pd.DataFrame): DataFrame containing the target/label variable.
        selected_features (Dict[str, Optional[str]]): Mapping of logical feature names to
                                       actual column names in the input data.

    Returns:
        Tuple[float, str, Dict[str, Any]]:
            - Overall representativity score (0.0-1.0), averaged across features.
            - Combined explanation string summarizing feature-level results.
            - Dictionary of detailed per-feature results.
    """

    # If all selected features are None, nothing to analyze
    non_valid_features = (
        0,
        "No valid features found for representativity analysis",
        {},
    )

    if all(x is None for x in selected_features.values()):
        return non_valid_features

    # Initialize empty merged analysis results
    feature_scores = {}
    feature_explanations = {}
    detailed_results = {}

    # Union of all column names across tabular and target data
    all_columns = set(tabular_data.columns) | set(target_data.columns)

    for feat, column_name in selected_features.items():

        # Skip if no column name was assigned for this feature
        if not column_name:
            continue

        # If the column exists in either tabular_data or target_data
        if column_name in all_columns:

            if feat == "target":
                # Analyze target as a categorical variable
                feature_score, feature_explanation, feature_details = (
                    _analyze_categorical_representativity(
                        data=target_data, column=column_name, feature_name=column_name
                    )
                )

            elif feat in {"gender", "subpopulation"}:
                # Analyze categorical demographic features
                feature_score, feature_explanation, feature_details = (
                    _analyze_categorical_representativity(
                        data=tabular_data, column=column_name, feature_name=feat
                    )
                )

            elif feat == "age":
                # Analyze age using age-binned representativity
                feature_score, feature_explanation, feature_details = (
                    _analyze_age_representativity(
                        data=tabular_data, column=column_name, feature_name=feat
                    )
                )
            else:
                # Unsupported feature key
                feature_score, feature_explanation, feature_details = (
                    0,
                    f"No valid feature with the name {feat} for representativity analysis",
                    {},
                )
        else:
            # Column not found in provided data
            feature_score, feature_explanation, feature_details = (
                0,
                f"{feat}: Column '{column_name}' not found in clinical data",
                {},
            )

        # Save results for this feature
        feature_scores[feat] = feature_score
        feature_explanations[feat] = feature_explanation

        detailed_results[feat] = {
            "score": feature_score,
            "explanation": feature_explanation,
            "details": feature_details,
        }

    # Return in case no features were processed
    if not feature_scores:
        return non_valid_features

    # Compute overall score as the average of all feature scores
    overall_score = sum(feature_scores.values()) / len(feature_scores)
    # Combine explanations into one summary
    merged_explanation = f"Multi-feature representativity analysis. {'; '.join(feature_explanations.values())}."

    return overall_score, merged_explanation, detailed_results


def check_metadata_granularity_tabular(
    tabular_data: pd.DataFrame,
    target_data: pd.DataFrame,
    metadata: Optional[pd.DataFrame] = None,
) -> Tuple[int, str]:
    """Check metadata granularity for a given dataset and target.
    This function evaluates whether metadata is provided for a dataset.

    Args:
        tabular_data (pd.DataFrame): The main tabular dataset under analysis.
        target_data (pd.DataFrame): The target dataset (labels, outcomes, etc.).
        metadata (Optional[pd.DataFrame]): Metadata describing the dataset. Defaults to None.

    Returns:
        Tuple[int, str]:
            - An integer status code (1 indicates missing metadata).
            - A descriptive message about metadata availability.
    """

    # If metadata DataFrame is not provided, return score 1 with a message
    if metadata is None:
        return 1, "No metadata available for Use Case 4."


def check_accuracy_tabular(
    data: pd.DataFrame, expected_ranges: Dict[str, Tuple[float, float]]
) -> Tuple[float, str]:
    """Check how many values in specified columns fall within expected numeric ranges.

    Args:
        data (pd.DataFrame): The tabular dataset containing numeric columns to check.
        expected_ranges (Dict[str, Tuple[float, float]]): Dictionary where keys are column names
            and values are tuples specifying the acceptable (min, max) range for that column.

    Returns:
        Tuple[float, str]:
            - Accuracy ratio (0.0-1.0) of values within acceptable ranges.
            - Detailed message summarizing results and any columns with out-of-range values.
    """

    # If no expected ranges provided, return "perfect" score with message
    if not expected_ranges:
        return (
            1.0,
            "No columns selected for accuracy check. Cannot calculate accuracy metric.",
        )

    total_values_checked = 0  # Total numeric values across all columns
    values_within_range = 0  # Count of values falling within the specified ranges
    columns_checked = 0  # Number of columns successfully checked
    error_details = []  # Collect messages for columns with out-of-range values

    for column, acceptable_values in expected_ranges.items():
        if column in data.columns:

            # Count non-NaN values in this column
            num_column_values = data[column].count().__int__()

            # Count values within the specified range
            column_acceptable_values = (
                data[column]
                .between(acceptable_values[0], acceptable_values[1])
                .sum()
                .__int__()
            )

            # Track number of values checked and within the acceptable range
            columns_checked += 1
            total_values_checked += num_column_values
            values_within_range += column_acceptable_values

            # Track number of values outside the acceptable range
            outside_acceptable = num_column_values - column_acceptable_values
            if outside_acceptable > 0:
                error_details.append(
                    f"{column}: {outside_acceptable}/{num_column_values} values not in acceptable range {acceptable_values}"
                )

    # Handle edge cases where no columns were checked
    if columns_checked == 0:
        return (
            1.0,
            "Selected columns not found tabular data. Cannot calculate accuracy.",
        )

    if total_values_checked == 0:
        return 1.0, "No numeric values found in tabular data columns."

    # Compute accuracy ratio
    accuracy_ratio = values_within_range / total_values_checked

    # Prepare detailed message including any errors
    if error_details:
        details_str = ", ".join(error_details)
        return (
            accuracy_ratio,
            f"Checked {columns_checked} column(s). {values_within_range}/{total_values_checked} values ({accuracy_ratio:.1%}) are within the acceptable ranges. Issues: {details_str}",
        )
    else:
        return (
            accuracy_ratio,
            f"Checked {columns_checked} column(s). All {total_values_checked} values are within the acceptable ranges.",
        )


def check_coherence_tabular(
    data: pd.DataFrame, exp_col_types: Dict[str, str]
) -> Tuple[float, str]:
    """
    Check if the columns in a tabular DataFrame conform to expected data types.
    Validates each specified column against its expected type ("numeric" or "categorical")
    and returns a coherence score and a descriptive message.

    Args:
        data (pd.DataFrame): The tabular data to check.
        exp_col_types (Dict[str, str]): A dictionary mapping column names to expected types.

    Returns:
        Tuple[float, str]:
            - Coherence ratio (0.0 to 1.0) indicating the fraction of consistent columns.
            - Message describing the check result and any inconsistencies.
    """

    # If no expected types provided, cannot check coherence
    if not exp_col_types:
        return 1.0, "No column type specifications provided. Cannot check coherence."

    total_features_checked = 0
    consistent_features = 0
    inconsistent_details = []

    # Set a generic number of categories for categorical columns
    number_of_expected_categories = 20

    for column, expected_type in exp_col_types.items():

        column_consistent = True

        # Skip columns not present in the DataFrame
        if not column in data.columns:
            continue

        if expected_type == "numeric":

            # Check if column is numeric type
            if not pd.api.types.is_numeric_dtype(data[column]):
                column_consistent = False

                # Count non-numeric entries (excluding NaNs)
                non_numeric_count = (
                    data[column].apply(pd.to_numeric, errors="coerce").isna().sum()
                    - data[column].isna().sum()
                ).__int__()

                inconsistent_details.append(
                    f"{column}: Expected numeric but contains {non_numeric_count} non-numeric values."
                )

        elif expected_type == "categorical":
            # For categorical, check if the number of unique values is reasonable
            unique_vals = data[column].dropna().nunique()

            if unique_vals > number_of_expected_categories:
                column_consistent = False

                inconsistent_details.append(
                    f"{column}: Expected categorical but has {unique_vals} unique values (too many for UC4 categorical)."
                )

        total_features_checked += 1

        if column_consistent:
            consistent_features += 1

    # If no expected features were found in the DataFrame
    if total_features_checked == 0:
        return 1.0, "Expected features not found in tabular data columns."

    # Calculate coherence as ratio as fraction of consistent features
    coherence_ratio = consistent_features / total_features_checked

    if inconsistent_details:
        details = ", ".join(inconsistent_details)
        return (
            coherence_ratio,
            f"Checked {total_features_checked} feture(s). {consistent_features}/{total_features_checked} features have consistent data types. Issues: {details}",
        )
    else:
        return (
            coherence_ratio,
            f"Checked {total_features_checked} feture(s). All {total_features_checked} features have consistent data types.",
        )


def check_semantic_coherence_tabular(data: pd.DataFrame) -> Tuple[float, str]:
    """
    Check the semantic coherence of a tabular DataFrame by verifying column name uniqueness.

    Args:
        data (pd.DataFrame): The DataFrame to check.

    Returns:
        Tuple[float, str]:
            - Coherence ratio (0.0 to 1.0) representing the fraction of unique columns.
            - Message describing whether duplicate column names were found.
    """

    # Total number of columns in the DataFrame
    num_columns = data.columns.__len__()

    # If no columns exist, return perfect score (nothing to check)
    if num_columns == 0:
        return 1.0, "No columns found in tabular data."

    # Identify columns that appear more than once
    duplicate_columns = [
        name for name, count in Counter(list(data.columns)).items() if count > 1
    ]
    num_duplicates = len(duplicate_columns)

    # Count of unique columns
    unique_columns = num_columns - num_duplicates

    # Prepare duplicate details for message
    if num_duplicates > 0:
        duplicate_list = [f"'{name}'" for name in duplicate_columns]
        duplicate_details = ", ".join(duplicate_list)

    # Coherence ratio = fraction of unique columns
    coherence_ratio = unique_columns / num_columns

    if num_duplicates == 0:
        return (
            1.0,
            f"No duplicate column names found. All column names are unique.",
        )
    else:
        return (
            coherence_ratio,
            f"Found duplicate column names. Duplicates: {duplicate_details}. "
            f"Overall coherence: {unique_columns}/{num_columns} unique columns.",
        )


def check_completeness_tabular(data: pd.DataFrame) -> Tuple[float, str]:
    """
    Evaluate completeness of a tabular dataset by checking for missing values (NaNs).

    Args:
        data (pd.DataFrame): The input tabular dataset.

    Returns:
        Tuple[float, str]:
            - A completeness ratio (float between 0 and 1).
            - A descriptive message summarizing completeness and any missing values per column.
    """

    # Total number of values in the dataframe
    total_values = data.size

    # Handle edge case, empty dataframe
    if total_values == 0:
        return 1.0, "No data to check."

    # Count total missing values across all columns
    total_missing_values = data.isna().sum().sum().__int__()

    # Compute number of non-missing values and completeness ratio
    complete_values = total_values - total_missing_values
    completeness_ratio = complete_values / total_values

    if total_missing_values > 0:
        # Collect missing value details per column
        missing_details = []

        for col in data.columns:
            col_missing = int(data[col].isna().sum())
            col_total = int(data[col].size)

            if col_missing > 0:
                col_percent = (col_missing / col_total) * 100
                missing_details.append(f"{col}: {col_missing} ({col_percent:.1f}%)")

        # Join details into a readable string
        missing_details = ", ".join(missing_details)

        return (
            completeness_ratio,
            f"Data completeness: {complete_values}/{total_values} ({completeness_ratio:.1%}). Missing values by column: {missing_details}",
        )
    else:
        # No missing values, perfect completeness
        return (
            completeness_ratio,
            f"No missing values found in tabular data. Data is 100% complete.",
        )


def check_relational_consistency_tabular(data: pd.DataFrame) -> Tuple[float, str]:
    """
    Check relational consistency of tabular data by identifying duplicate rows.

    Args:
        data (pd.DataFrame): Input tabular dataset to check.

    Returns:
        Tuple[float, str]:
            - Consistency ratio (float between 0 and 1), where 1.0 means all rows are unique.
            - A descriptive message summarizing whether duplicates were found.
    """

    # Total number of rows in dataset
    total_rows = len(data)

    # Handle case of empty dataframe
    if total_rows == 0:
        return 1.0, "No data to check."

    # Count unique rows by dropping duplicates
    unique_rows = len(data.drop_duplicates())

    # Ratio of unique rows to total rows
    consistency_ratio = unique_rows / total_rows

    # Number of duplicate rows
    duplicate_rows = total_rows - unique_rows

    if duplicate_rows > 0:
        # Boolean mask of duplicated rows
        duplicated_mask = data.duplicated(keep="first")
        # Get indices of duplicated rows
        duplicated_indices = data[duplicated_mask].index.tolist()

        # Percentage of duplicates relative to dataset size
        duplicate_perc = duplicate_rows / total_rows

        # Show at most 10 duplicate indices for readability
        if len(duplicated_indices) > 10:
            indices_str = str(duplicated_indices[:10])[:-1] + ", ...]"
        else:
            indices_str = str(duplicated_indices)

        # Details for duplicate rows in text format
        duplicate_details = f"{duplicate_rows} duplicate rows out of {total_rows} ({duplicate_perc:.1%}), indices: {indices_str}"

        return (
            consistency_ratio,
            f"Found duplicate rows in tabular data ({total_rows} rows checked in total). {unique_rows} rows are unique. Duplicates details: {duplicate_details}.",
        )

    else:
        # All rows are unique, perfect relational consistency
        return (
            consistency_ratio,
            f"No duplicate rows found in tabular data. All {total_rows} rows are unique.",
        )


def run_all_checks_tabular(
    tabular_data: pd.DataFrame,
    target_data: pd.DataFrame,
    uc_conf: Dict[str, Any],
    selected_feat: Dict[str, Optional[str]],
) -> Dict[str, Tuple[float, str]]:
    """
    Run all tabular data quality checks and return results in a dictionary.

    Args:
        tabular_data (pd.DataFrame): Main dataset (X) to validate.
        target_data (pd.DataFrame): Target data (Y) for some checks.
        uc_conf (Dict[str, Any]): Use case configuration dictionary.
        selected_features (Dict[str, Optional[str]]):
            Dictionary of selected features to include in the population representativity check.

    Returns:
        Dict[str, Tuple[float, str]]:
            A dictionary mapping each check name to a tuple:
            - Score (float between 0 and 1).
            - Descriptive message string.
    """

    # Initialize results dict
    results = {}

    # Check how well the dataset represents population across selected features
    results["population_representativity"] = check_population_representativity_tabular(
        tabular_data=tabular_data,
        target_data=target_data,
        selected_features=selected_feat,
    )

    # Check metadata availability and granularity
    results["metadata_granularity"] = check_metadata_granularity_tabular(
        tabular_data=tabular_data, target_data=target_data
    )

    # Check whether values fall within expected ranges
    results["accuracy"] = check_accuracy_tabular(
        data=tabular_data, expected_ranges=uc_conf.get("expected_ranges")
    )

    # Check consistency of column data types
    results["coherence"] = check_coherence_tabular(
        data=tabular_data, exp_col_types=uc_conf.get("column_types")
    )

    # Check for semantic issues like duplicate column names
    results["semantic_coherence"] = check_semantic_coherence_tabular(data=tabular_data)

    # Check for missing values and completeness of data
    results["completeness"] = check_completeness_tabular(data=tabular_data)

    # Check for duplicate rows to ensure relational consistency
    results["relational_consistency"] = check_relational_consistency_tabular(
        data=tabular_data
    )

    return results
