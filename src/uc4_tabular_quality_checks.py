import pandas as pd
from typing import Tuple, Dict, Any, Optional


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
    explanation = f"Balance score {score:.3f} (ideal: {ideal_proportion:.1%} each), Distribution: {details_str}"

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
        selected_features (Dict[str]): Mapping of logical feature names to
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


def check_metadata_granularity(
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


def run_all_checks_tabular(tabular_data, target_data, uc_conf, selected_feat):

    results = {}

    results["population_representativity"] = check_population_representativity_tabular(
        tabular_data=tabular_data,
        target_data=target_data,
        selected_features=selected_feat,
    )

    results["metadata_granularity"] = check_metadata_granularity(
        tabular_data=tabular_data, target_data=target_data
    )

    return results
