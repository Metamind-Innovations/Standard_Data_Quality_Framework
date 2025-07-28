import pandas as pd


def check_population_representativity_time_series(patient_data, selected_features=None):
    """
    Check population representativity in time series patient data based on demographic features.

    **Population Representativity Check**

    Population representativity refers to the degree to which the data adequately
    represent the population in question. For time series patient data, this is measured
    by checking that each demographic subgroup (age groups, gender groups) contains
    patients with all diabetic status values (0, 1, 2).

    **SDQF Rating Criteria:**

    * All subgroups have all diabetic status values: Rating 5/5 (excellent representativity)
    * Some subgroups missing diabetic status values: Rating 1-4/5 (based on coverage)
    * Most subgroups missing diabetic status values: Rating 1/5 (poor representativity)

    The function analyzes age groups (0-20, 20-40, 40-60, 60-80, 80+) and gender groups
    to ensure each contains patients with diabetic status 0, 1, and 2.

    Args:
        patient_data: Dictionary containing processed patient data with demographics
        selected_features: Dictionary of selected features (ignored, uses age and gender only)

    Returns:
        Tuple of (score, explanation, detailed_results) where score is 0-1 and
        explanation describes the analysis
    """
    if not patient_data:
        return 0, "No patient data found", {}

    demographic_data = []
    for patient_id, data in patient_data.items():
        demographics = data.get("demographics", {})
        if demographics:
            patient_record = demographics.copy()
            patient_record["PatientID"] = patient_id
            demographic_data.append(patient_record)

    if not demographic_data:
        return 0, "No demographic data found for patients", {}

    clinical_data = pd.DataFrame(demographic_data)

    if "diabeticStatus" not in clinical_data.columns:
        return (
            0,
            "No diabeticStatus information found for representativity analysis",
            {},
        )

    valid_data = clinical_data.dropna(subset=["diabeticStatus"])
    if len(valid_data) == 0:
        return 0, "No patients with valid diabeticStatus found", {}

    diabetic_status_values = set(valid_data["diabeticStatus"].unique())

    detailed_results = {}
    subgroup_scores = []
    explanations = []

    # Analyze age groups
    if "age" in valid_data.columns:
        age_score, age_explanation, age_details = _analyze_age_representativity_fixed(
            valid_data, diabetic_status_values
        )
        subgroup_scores.append(age_score)
        explanations.append(f"Age groups: {age_explanation}")
        detailed_results["age_groups"] = age_details

    # Analyze gender groups
    if "gender" in valid_data.columns:
        gender_score, gender_explanation, gender_details = (
            _analyze_gender_representativity_fixed(valid_data, diabetic_status_values)
        )
        subgroup_scores.append(gender_score)
        explanations.append(f"Gender groups: {gender_explanation}")
        detailed_results["gender_groups"] = gender_details

    if not subgroup_scores:
        return 0, "No valid demographic features (age, gender) found for analysis", {}

    overall_score = sum(subgroup_scores) / len(subgroup_scores)

    total_patients = len(valid_data)

    diabetic_counts = valid_data["diabeticStatus"].value_counts()
    diabetic_labels = {0: "Healthy", 1: "Type 1 Diabetes", 2: "Type 2 Diabetes"}

    diabetic_distribution_formatted = []
    for status, count in diabetic_counts.items():
        label = diabetic_labels.get(int(status), f"Status {status}")
        diabetic_distribution_formatted.append(f"{label}: {int(count)}")

    diabetic_distribution_str = "{" + ", ".join(diabetic_distribution_formatted) + "}"

    combined_explanation = (
        f"Population representativity across demographic subgroups (score: {overall_score:.3f}). "
        f"Total patients: {total_patients}, "
        f"Diabetic status distribution: {diabetic_distribution_str}. "
        f"{'; '.join(explanations)}."
    )

    formatted_detailed_results = {}
    if "age_groups" in detailed_results:
        formatted_detailed_results["age"] = {
            "score": detailed_results["age_groups"]["diversity_ratio"],
            "explanation": f"Age groups: {detailed_results['age_groups']['groups_with_full_diversity']}/{detailed_results['age_groups']['total_groups']} have all diabetic status values",
            "details": detailed_results["age_groups"],
        }

    if "gender_groups" in detailed_results:
        formatted_detailed_results["gender"] = {
            "score": detailed_results["gender_groups"]["diversity_ratio"],
            "explanation": f"Gender groups: {detailed_results['gender_groups']['groups_with_full_diversity']}/{detailed_results['gender_groups']['total_groups']} have all diabetic status values",
            "details": detailed_results["gender_groups"],
        }

    return overall_score, combined_explanation, formatted_detailed_results


def _analyze_age_representativity_fixed(valid_data, diabetic_status_values):
    """
    Analyze the representativity of diabetic status values across predefined age groups.

    This function evaluates whether each age group (bins: 0-20, 20-40, 40-60, 60-80, 80+)
    contains all possible diabetic status values provided in `diabetic_status_values`.
    It computes a diversity ratio, which is the proportion of age groups that have
    full representation of all diabetic status values.

    Args:
        valid_data (pd.DataFrame): DataFrame containing at least 'age' and 'diabeticStatus' columns.
        diabetic_status_values (list or set): The set of expected diabetic status values (e.g., [0, 1, 2]).

    Returns:
        tuple:
            - diversity_ratio (float): Proportion of age groups with all diabetic status values present.
            - explanation (str): Human-readable summary of the representativity analysis.
            - details (dict): Detailed results for each age group, including patient counts,
                              present diabetic status values, counts, and diversity metrics.

    If no valid age data is found, or if an error occurs, returns a score of 0 and an explanatory message.
    """
    # Filter patients with valid age
    age_data = valid_data.copy()
    age_data["age"] = pd.to_numeric(age_data["age"], errors="coerce")
    age_data = age_data.dropna(subset=["age"])

    if len(age_data) == 0:
        return 0, "No patients with valid age data", {}

    age_bins = [0, 20, 40, 60, 80, 120]
    age_labels = ["0-20", "20-40", "40-60", "60-80", "80+"]

    try:
        age_data["age_group"] = pd.cut(
            age_data["age"], bins=age_bins, labels=age_labels, include_lowest=True
        )
        age_data = age_data.dropna(subset=["age_group"])

        if len(age_data) == 0:
            return 0, "No patients with valid age groups", {}

        # Check each age group for diabetic status diversity
        age_group_results = {}
        groups_with_full_diversity = 0
        total_age_groups = 0

        diabetic_labels = {0: "Healthy", 1: "Type 1 Diabetes", 2: "Type 2 Diabetes"}

        for age_group in age_labels:
            group_data = age_data[age_data["age_group"] == age_group]
            if len(group_data) > 0:
                total_age_groups += 1
                group_diabetic_values = set(group_data["diabeticStatus"].unique())
                has_full_diversity = len(group_diabetic_values) == len(
                    diabetic_status_values
                )

                if has_full_diversity:
                    groups_with_full_diversity += 1

                diabetic_counts_raw = group_data["diabeticStatus"].value_counts()
                diabetic_counts_formatted = {}
                for status, count in diabetic_counts_raw.items():
                    label = diabetic_labels.get(int(status), f"Status {status}")
                    diabetic_counts_formatted[label] = int(count)

                age_group_results[age_group] = {
                    "patient_count": len(group_data),
                    "diabetic_status_present": [
                        diabetic_labels.get(int(s), f"Status {s}")
                        for s in group_diabetic_values
                    ],
                    "diabetic_status_counts": diabetic_counts_formatted,
                    "has_full_diversity": has_full_diversity,
                    "diversity_ratio": len(group_diabetic_values)
                    / len(diabetic_status_values),
                }

        if total_age_groups == 0:
            return 0, "No age groups with patients found", {}

        diversity_ratio = groups_with_full_diversity / total_age_groups

        explanation = (
            f"{groups_with_full_diversity}/{total_age_groups} age groups have all diabetic status values "
            f"(diversity ratio: {diversity_ratio:.2f})"
        )

        details = {
            "diversity_ratio": diversity_ratio,
            "groups_with_full_diversity": groups_with_full_diversity,
            "total_groups": total_age_groups,
            "age_group_details": age_group_results,
            "expected_diabetic_values": [
                diabetic_labels.get(int(s), f"Status {s}")
                for s in diabetic_status_values
            ],
        }

        return diversity_ratio, explanation, details

    except Exception as e:
        return 0, f"Error analyzing age groups: {str(e)}", {}


def _analyze_gender_representativity_fixed(valid_data, diabetic_status_values):
    """
    Analyze the representativity of gender groups with respect to diabetic status diversity.

    This function evaluates whether each gender group in the provided data contains all expected diabetic status values.
    It calculates the diversity ratio, which is the proportion of gender groups that have all diabetic status categories
    present. The function also provides detailed statistics for each gender group, including the count of patients,
    which diabetic statuses are present, counts for each status, and whether the group has full diversity.

    Args:
        valid_data (pd.DataFrame): DataFrame containing patient data with at least 'gender' and 'diabeticStatus' columns.
        diabetic_status_values (list or set): The set of expected diabetic status values (e.g., [0, 1, 2]).

    Returns:
        tuple:
            - diversity_ratio (float): Proportion of gender groups with all diabetic status values present.
            - explanation (str): Human-readable summary of the diversity analysis.
            - details (dict): Detailed results for each gender group, including counts and diversity information.

    If no patients with valid gender data are found, returns (0, explanation, {}).
    """
    # Filter patients with valid gender
    gender_data = valid_data.dropna(subset=["gender"])

    if len(gender_data) == 0:
        return 0, "No patients with valid gender data", {}

    gender_groups = gender_data["gender"].unique()
    gender_group_results = {}
    groups_with_full_diversity = 0
    total_gender_groups = len(gender_groups)

    diabetic_labels = {0: "Healthy", 1: "Type 1 Diabetes", 2: "Type 2 Diabetes"}

    for gender in gender_groups:
        group_data = gender_data[gender_data["gender"] == gender]
        group_diabetic_values = set(group_data["diabeticStatus"].unique())
        has_full_diversity = len(group_diabetic_values) == len(diabetic_status_values)

        if has_full_diversity:
            groups_with_full_diversity += 1

        diabetic_counts_raw = group_data["diabeticStatus"].value_counts()
        diabetic_counts_formatted = {}
        for status, count in diabetic_counts_raw.items():
            label = diabetic_labels.get(int(status), f"Status {status}")
            diabetic_counts_formatted[label] = int(count)

        gender_group_results[gender] = {
            "patient_count": len(group_data),
            "diabetic_status_present": [
                diabetic_labels.get(int(s), f"Status {s}")
                for s in group_diabetic_values
            ],
            "diabetic_status_counts": diabetic_counts_formatted,
            "has_full_diversity": has_full_diversity,
            "diversity_ratio": len(group_diabetic_values) / len(diabetic_status_values),
        }

    diversity_ratio = (
        groups_with_full_diversity / total_gender_groups
        if total_gender_groups > 0
        else 0
    )

    explanation = (
        f"{groups_with_full_diversity}/{total_gender_groups} gender groups have all diabetic status values "
        f"(diversity ratio: {diversity_ratio:.2f})"
    )

    details = {
        "diversity_ratio": diversity_ratio,
        "groups_with_full_diversity": groups_with_full_diversity,
        "total_groups": total_gender_groups,
        "gender_group_details": gender_group_results,
        "expected_diabetic_values": [
            diabetic_labels.get(int(s), f"Status {s}") for s in diabetic_status_values
        ],
    }

    return diversity_ratio, explanation, details


def check_metadata_granularity_time_series(patient_data):
    """
    Check metadata granularity for time series patient data.

    **Metadata Granularity Check**

    Metadata granularity refers to the availability, comprehensiveness and level of detail
    of metadata that help users understand the data being used. For time series patient data,
    this is measured by checking the ratio of patients with complete demographic metadata
    to total patients.

    **SDQF Rating Criteria:**

    * Patients with complete metadata ratio ≤0.2: Rating 1/5 (poor metadata coverage)
    * Patients with complete metadata ratio ≥0.8: Rating 5/5 (excellent metadata coverage)

    The function considers a patient to have complete metadata if they have at least 3 out of 4
    key demographic fields: age, gender, diabeticStatus, and weight (excluding hospitalID).

    Args:
        patient_data: Dictionary containing processed patient data

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not patient_data:
        return 0, "No patient data provided for metadata granularity check"

    total_patients = len(patient_data)
    patients_with_complete_metadata = 0

    key_demographic_fields = ["age", "gender", "diabeticStatus", "weight"]

    for patient_id, data in patient_data.items():
        demographics = data.get("demographics", {})

        if not demographics:
            continue

        fields_present = 0
        for field in key_demographic_fields:
            if field in demographics and demographics[field] is not None:
                fields_present += 1

        # Consider metadata complete if at least 3 out of 4 key fields are present
        if fields_present >= 3:
            patients_with_complete_metadata += 1

    ratio = (
        patients_with_complete_metadata / total_patients if total_patients > 0 else 0
    )

    return (
        ratio,
        f"Patients with complete demographic metadata: {patients_with_complete_metadata}/{total_patients} ({ratio:.1%}). "
        f"Complete metadata requires at least 3 out of 4 key fields: {', '.join(key_demographic_fields)}.",
    )


def check_accuracy_time_series(patient_data, expected_ranges=None):
    """
    Check accuracy of values in time series data against expected ranges.

    **Accuracy Check**

    Accuracy refers to the degree to which data correctly describes what it was designed
    to measure (the "real world" entity). For time series patient data, this is measured
    by checking that demographic and time series values fall within medically reasonable
    ranges or acceptable value lists.

    **SDQF Rating Criteria:**

    * Values within acceptable ranges ≤20%: Rating 1/5 (poor accuracy)
    * Values within acceptable ranges ≥80%: Rating 5/5 (excellent accuracy)

    The function always checks age (0-120 years) and additionally checks any field selected
    from "Select Additional Columns for Validation" (gender, diabeticStatus, or weight)
    against their respective acceptable ranges.

    Args:
        patient_data: Dictionary containing processed patient data
        expected_ranges: Dictionary mapping field names to acceptable values or ranges

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    # If no expected ranges provided, return perfect score
    if not expected_ranges:
        return (
            1.0,
            "No fields selected for accuracy check. Cannot calculate accuracy metric.",
        )

    if not patient_data:
        return 1.0, "No patient data provided for accuracy check."

    total_values_checked = 0
    values_within_range = 0
    error_details = []
    fields_checked = 0

    # Always check age, and only check other fields if they are specifically selected in expected_ranges
    fields_to_check = ["age"]  # Always check age

    for field_name in expected_ranges.keys():
        if field_name != "age" and field_name not in fields_to_check:
            fields_to_check.append(field_name)

    for field_name in fields_to_check:
        field_found = False
        field_total_values = 0
        field_acceptable_values = 0

        if field_name == "age":
            acceptable_values = expected_ranges.get("age", (0, 120))
        else:
            if field_name not in expected_ranges:
                continue
            acceptable_values = expected_ranges[field_name]

        for patient_id, data in patient_data.items():
            demographics = data.get("demographics", {})
            if field_name in demographics:
                field_found = True
                value = demographics[field_name]
                if value is not None:
                    field_total_values += 1

                    if _is_value_acceptable(value, acceptable_values):
                        field_acceptable_values += 1

        if field_found:
            fields_checked += 1
            total_values_checked += field_total_values
            values_within_range += field_acceptable_values

            outside_acceptable = field_total_values - field_acceptable_values
            if outside_acceptable > 0:
                error_details.append(
                    f"{field_name}: {outside_acceptable}/{field_total_values} values not in acceptable range {acceptable_values}"
                )

    if fields_checked == 0:
        return (
            1.0,
            "Selected fields not found in patient data. Cannot calculate accuracy.",
        )

    if total_values_checked == 0:
        return 1.0, "No values found in selected fields across patients."

    accuracy_ratio = values_within_range / total_values_checked

    if error_details:
        details_str = ", ".join(error_details)
        return (
            accuracy_ratio,
            f"Checked {fields_checked} field(s) across {len(patient_data)} patients. {values_within_range}/{total_values_checked} values ({accuracy_ratio:.1%}) are within acceptable ranges. Issues: {details_str}",
        )
    else:
        return (
            accuracy_ratio,
            f"Checked {fields_checked} field(s) across {len(patient_data)} patients. All {total_values_checked} values are within acceptable ranges.",
        )


def _is_value_acceptable(value, acceptable_criteria):
    """
    Check if a value meets the acceptable criteria.

    Args:
        value: The value to check
        acceptable_criteria: Can be:
            - List of acceptable values
            - Tuple of (min, max) for numeric ranges
            - Single value for exact match

    Returns:
        Boolean indicating if value is acceptable
    """
    try:
        if isinstance(acceptable_criteria, list):
            return value in acceptable_criteria
        elif isinstance(acceptable_criteria, tuple) and len(acceptable_criteria) == 2:
            min_val, max_val = acceptable_criteria
            numeric_value = float(value)
            return min_val <= numeric_value <= max_val
        else:
            return value == acceptable_criteria
    except (ValueError, TypeError):
        return value == acceptable_criteria


def check_coherence_time_series(patient_data, column_types=None):
    """
    Check coherence of data types in time series patient data against expected types.

    **Coherence Check**

    Coherence refers to the degree to which data types are consistent and logical
    throughout the dataset. For time series patient data, this is measured by checking
    that demographic and time series fields have consistent data types across all patients.

    **SDQF Rating Criteria:**

    * All specified fields have consistent data types: Rating 5/5 (excellent coherence)
    * Some fields have inconsistent data types: Rating 1-4/5 (based on consistency ratio)
    * Most fields have inconsistent data types: Rating 1/5 (poor coherence)

    Args:
        patient_data: Dictionary containing processed patient data
        column_types: Dictionary mapping field names to expected types ('numeric' or 'categorical')

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not column_types:
        return 1.0, "No field type specifications provided. Cannot check coherence."

    if not patient_data:
        return 1.0, "No patient data provided for coherence check."

    total_fields = len(column_types)
    consistent_fields = 0
    inconsistent_details = []

    for field_name, expected_type in column_types.items():
        field_found = False
        field_consistent = True
        total_non_numeric = 0
        total_unique_values = 0
        patients_with_field = 0
        all_values = []

        for patient_id, data in patient_data.items():
            demographics = data.get("demographics", {})
            time_series = data.get("time_series", {})

            field_value = None
            if field_name in demographics:
                field_value = demographics[field_name]
                field_found = True
            elif field_name in time_series:
                ts_values = time_series[field_name]
                if isinstance(ts_values, list):
                    extracted_values = []
                    for ts_entry in ts_values:
                        if isinstance(ts_entry, list) and len(ts_entry) >= 2:
                            value = ts_entry[1]
                            if value is not None:
                                extracted_values.append(value)
                    field_value = extracted_values
                else:
                    field_value = [ts_values] if ts_values is not None else []
                field_found = True

            if field_value is not None:
                patients_with_field += 1

                values_to_check = (
                    field_value if isinstance(field_value, list) else [field_value]
                )

                filtered_values = []
                for v in values_to_check:
                    if v is not None:
                        try:
                            if not pd.isna(v):
                                filtered_values.append(v)
                        except (TypeError, ValueError):
                            filtered_values.append(v)

                if filtered_values:
                    all_values.extend(filtered_values)

        if field_found and all_values:
            if expected_type == "numeric":
                try:
                    numeric_values = []
                    non_numeric_count = 0
                    for value in all_values:
                        try:
                            numeric_val = pd.to_numeric(value, errors="raise")
                            numeric_values.append(numeric_val)
                        except (ValueError, TypeError):
                            non_numeric_count += 1

                    if non_numeric_count > 0:
                        field_consistent = False
                        total_non_numeric = non_numeric_count

                except Exception:
                    field_consistent = False
                    total_non_numeric = len(all_values)

            elif expected_type == "categorical":
                # For categorical, check if it has a reasonable number of unique values
                unique_vals = len(set(all_values))
                total_unique_values = unique_vals
                if unique_vals > 50:
                    field_consistent = False

        if field_found:
            if field_consistent:
                consistent_fields += 1
            else:
                if expected_type == "numeric" and total_non_numeric > 0:
                    inconsistent_details.append(
                        f"{field_name}: Expected numeric but contains {total_non_numeric} non-numeric values across {patients_with_field} patients"
                    )
                elif expected_type == "categorical" and total_unique_values > 50:
                    inconsistent_details.append(
                        f"{field_name}: Expected categorical but has {total_unique_values} unique values (too many for categorical)"
                    )

    coherence_ratio = consistent_fields / total_fields if total_fields > 0 else 1.0

    if inconsistent_details:
        details = ", ".join(inconsistent_details)
        return (
            coherence_ratio,
            f"Checked {len(patient_data)} patients. {consistent_fields}/{total_fields} fields have consistent data types. Issues: {details}",
        )
    else:
        return (
            coherence_ratio,
            f"Checked {len(patient_data)} patients. All {total_fields} fields have consistent data types.",
        )


def check_semantic_coherence_time_series(patient_data):
    """
    Check semantic coherence of time series data by detecting duplicate columns with identical values.

    **Semantic Coherence Check**

    Semantic coherence refers to the logical consistency and meaningfulness of data
    structure and relationships. For time series patient data, this is measured by
    detecting duplicate columns/fields that contain identical values across all patients.

    **SDQF Rating Criteria:**

    * No duplicate columns found: Rating 5/5 (excellent semantic coherence)
    * Some duplicate columns found: Rating 1-4/5 (based on uniqueness ratio)
    * Many duplicate columns: Rating 1/5 (poor semantic coherence)

    Args:
        patient_data: Dictionary containing processed patient data

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not patient_data:
        return 1.0, "No patient data provided for semantic coherence check."

    total_patients = len(patient_data)
    duplicate_issues = []
    total_fields = 0
    total_unique_fields = 0

    # 1. Check demographic fields for duplicates (exact same values across all patients)
    demographic_field_values = {}

    all_demographic_fields = set()
    for patient_id, data in patient_data.items():
        demographics = data.get("demographics", {})
        if isinstance(demographics, dict):
            all_demographic_fields.update(demographics.keys())

    for field_name in all_demographic_fields:
        field_values = []
        for patient_id, data in patient_data.items():
            demographics = data.get("demographics", {})
            if isinstance(demographics, dict) and field_name in demographics:
                field_values.append(demographics[field_name])
            else:
                field_values.append(None)
        demographic_field_values[field_name] = tuple(field_values)

    demographic_duplicates = []
    demographic_fields_list = list(all_demographic_fields)

    for i, field1 in enumerate(demographic_fields_list):
        for j, field2 in enumerate(demographic_fields_list[i + 1 :], i + 1):
            if demographic_field_values[field1] == demographic_field_values[field2]:
                demographic_duplicates.append((field1, field2))

    if demographic_duplicates:
        duplicate_pairs = [
            f"'{pair[0]}' = '{pair[1]}'" for pair in demographic_duplicates
        ]
        duplicate_issues.append(f"Demographics: {', '.join(duplicate_pairs)}")

    total_fields += len(all_demographic_fields)
    duplicate_demo_count = len(demographic_duplicates)
    total_unique_fields += len(all_demographic_fields) - duplicate_demo_count

    # 2. Check time series fields for duplicates (identical patterns across all patients)
    timeseries_field_patterns = {}

    all_timeseries_fields = set()
    for patient_id, data in patient_data.items():
        time_series = data.get("time_series", {})
        if isinstance(time_series, dict):
            all_timeseries_fields.update(time_series.keys())

    for field_name in all_timeseries_fields:
        field_patterns = []
        for patient_id, data in patient_data.items():
            time_series = data.get("time_series", {})
            if isinstance(time_series, dict) and field_name in time_series:
                ts_data = time_series[field_name]
                if isinstance(ts_data, list):
                    pattern = tuple(
                        tuple(entry) if isinstance(entry, list) else entry
                        for entry in ts_data
                    )
                else:
                    pattern = ts_data
                field_patterns.append(pattern)
            else:
                field_patterns.append(None)
        timeseries_field_patterns[field_name] = tuple(field_patterns)

    timeseries_duplicates = []
    timeseries_fields_list = list(all_timeseries_fields)

    for i, field1 in enumerate(timeseries_fields_list):
        for j, field2 in enumerate(timeseries_fields_list[i + 1 :], i + 1):
            if timeseries_field_patterns[field1] == timeseries_field_patterns[field2]:
                timeseries_duplicates.append((field1, field2))

    if timeseries_duplicates:
        duplicate_pairs = [
            f"'{pair[0]}' = '{pair[1]}'" for pair in timeseries_duplicates
        ]
        duplicate_issues.append(f"Time series: {', '.join(duplicate_pairs)}")

    total_fields += len(all_timeseries_fields)
    duplicate_ts_count = len(timeseries_duplicates)
    total_unique_fields += len(all_timeseries_fields) - duplicate_ts_count

    if total_fields == 0:
        return 1.0, "No fields found in patient data."

    coherence_ratio = total_unique_fields / total_fields if total_fields > 0 else 1.0

    if not duplicate_issues:
        return (
            1.0,
            f"No duplicate columns found across {total_patients} patients. "
            f"All {len(all_demographic_fields)} demographic fields and {len(all_timeseries_fields)} time series fields are unique.",
        )
    else:
        issues_str = "; ".join(duplicate_issues)
        total_duplicates = duplicate_demo_count + duplicate_ts_count
        return (
            coherence_ratio,
            f"Found duplicate columns in patient data. {total_duplicates} duplicate field pairs detected. "
            f"Duplicates: {issues_str}. Overall coherence: {total_unique_fields}/{total_fields} unique fields ({coherence_ratio:.3f}).",
        )


def check_completeness_time_series(patient_data):
    """
    Check completeness of time series data by detecting missing values in demographics and time series.

    **Completeness Check**

    Completeness refers to the degree to which all required data is present and
    available for use. For time series patient data, this is measured by checking
    for missing values in both demographic fields and time series measurements.

    **SDQF Rating Criteria:**

    * Data completeness ≤20%: Rating 1/5 (poor completeness)
    * Data completeness ≥80%: Rating 5/5 (excellent completeness)

    Args:
        patient_data: Dictionary containing processed patient data

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not patient_data:
        return 1.0, "No patient data provided for completeness check."

    total_values = 0
    missing_values = 0
    missing_by_field = {}

    # 1. Check demographic fields completeness
    all_demographic_fields = set()
    for patient_id, data in patient_data.items():
        demographics = data.get("demographics", {})
        if isinstance(demographics, dict):
            all_demographic_fields.update(demographics.keys())

    for field_name in all_demographic_fields:
        field_total_values = 0
        field_missing_values = 0

        for patient_id, data in patient_data.items():
            demographics = data.get("demographics", {})
            if isinstance(demographics, dict):
                field_total_values += 1

                if field_name not in demographics:
                    field_missing_values += 1
                else:
                    value = demographics[field_name]
                    if (
                        value is None
                        or (isinstance(value, str) and value.strip() == "")
                        or pd.isna(value)
                        or value == -1
                    ):
                        field_missing_values += 1

        total_values += field_total_values
        missing_values += field_missing_values

        if field_missing_values > 0:
            missing_by_field[field_name] = {
                "missing": field_missing_values,
                "total": field_total_values,
            }

    # 2. Check time series fields completeness
    all_timeseries_fields = set()
    for patient_id, data in patient_data.items():
        time_series = data.get("time_series", {})
        if isinstance(time_series, dict):
            all_timeseries_fields.update(time_series.keys())

    nutrition_fields = {"enteral_nutrition", "parenteral_nutrition"}
    regular_ts_fields = all_timeseries_fields - nutrition_fields

    for field_name in regular_ts_fields:
        field_total_data_points = 0
        field_missing_data_points = 0

        for patient_id, data in patient_data.items():
            time_series = data.get("time_series", {})
            if isinstance(time_series, dict):
                if field_name not in time_series:
                    # Patient missing entire time series for this field
                    # This counts as 1 missing data point for this patient
                    field_total_data_points += 1
                    field_missing_data_points += 1
                else:
                    ts_data = time_series[field_name]
                    if isinstance(ts_data, list):
                        for ts_entry in ts_data:
                            field_total_data_points += 1
                            if not isinstance(ts_entry, list) or len(ts_entry) < 2:
                                field_missing_data_points += 1
                            else:
                                timestamp, value = ts_entry[0], ts_entry[1]
                                if timestamp is None or value is None or pd.isna(value):
                                    field_missing_data_points += 1
                    else:
                        field_total_data_points += 1
                        if ts_data is None or pd.isna(ts_data):
                            field_missing_data_points += 1

        total_values += field_total_data_points
        missing_values += field_missing_data_points

        if field_missing_data_points > 0:
            missing_by_field[field_name] = {
                "missing": field_missing_data_points,
                "total": field_total_data_points,
            }

    # Special handling for nutrition fields - check for paired timestamps
    if nutrition_fields.intersection(all_timeseries_fields):
        enteral_total_expected = 0
        enteral_missing = 0
        parenteral_total_expected = 0
        parenteral_missing = 0

        for patient_id, data in patient_data.items():
            time_series = data.get("time_series", {})
            if isinstance(time_series, dict):
                enteral_data = time_series.get("enteral_nutrition", [])
                parenteral_data = time_series.get("parenteral_nutrition", [])

                enteral_timestamps = set()
                parenteral_timestamps = set()

                if isinstance(enteral_data, list):
                    for entry in enteral_data:
                        if isinstance(entry, list) and len(entry) >= 2:
                            enteral_timestamps.add(entry[0])

                if isinstance(parenteral_data, list):
                    for entry in parenteral_data:
                        if isinstance(entry, list) and len(entry) >= 2:
                            parenteral_timestamps.add(entry[0])

                all_nutrition_timestamps = enteral_timestamps.union(
                    parenteral_timestamps
                )

                for timestamp in all_nutrition_timestamps:
                    enteral_total_expected += 1
                    parenteral_total_expected += 1

                    if timestamp not in enteral_timestamps:
                        enteral_missing += 1
                    if timestamp not in parenteral_timestamps:
                        parenteral_missing += 1

        total_values += enteral_total_expected + parenteral_total_expected
        missing_values += enteral_missing + parenteral_missing

        if enteral_missing > 0:
            missing_by_field["enteral nutrition rate"] = {
                "missing": enteral_missing,
                "total": enteral_total_expected,
            }

        if parenteral_missing > 0:
            missing_by_field["parenteral nutrition rate"] = {
                "missing": parenteral_missing,
                "total": parenteral_total_expected,
            }

    if total_values == 0:
        return 1.0, "No data to check."

    complete_values = total_values - missing_values
    completeness_ratio = complete_values / total_values

    missing_fields = {
        field: data for field, data in missing_by_field.items() if data["missing"] > 0
    }

    if len(missing_fields) > 0:
        details = []
        for field, data in missing_fields.items():
            field_percent = (data["missing"] / data["total"]) * 100
            details.append(f"{field}: {data['missing']} ({field_percent:.1f}%)")
        details_str = ", ".join(details)
        return (
            completeness_ratio,
            f"Data completeness across {len(patient_data)} patients: {complete_values}/{total_values} ({completeness_ratio:.1%}). "
            f"Missing values by field: {details_str}",
        )
    else:
        return (
            completeness_ratio,
            f"No missing values found across {len(patient_data)} patients. Data is 100% complete.",
        )


def check_relational_consistency_time_series(patient_data):
    """
    Check relational consistency of time series data by detecting duplicate patients.

    **Relational Consistency Check**

    Relational consistency refers to the logical relationships and constraints that
    should hold between data elements. For time series patient data, this is measured
    by detecting duplicate patients based on identical demographic profiles and
    identical time series patterns.

    **SDQF Rating Criteria:**

    * No duplicate patients found: Rating 5/5 (excellent relational consistency)
    * Some duplicate patients found: Rating 1-4/5 (based on uniqueness ratio)
    * Many duplicate patients: Rating 1/5 (poor relational consistency)

    Args:
        patient_data: Dictionary containing processed patient data

    Returns:
        Tuple of (score, explanation) where score is 0-1 and explanation describes the analysis
    """
    if not patient_data:
        return 1.0, "No patient data provided for relational consistency check."

    total_patients = len(patient_data)
    duplicate_details = []

    # 1. Check for demographic duplicates (patients with identical demographic information)
    demographic_duplicates = 0
    demographic_patterns = {}

    for patient_id, data in patient_data.items():
        demographics = data.get("demographics", {})
        if isinstance(demographics, dict):
            demo_pattern = tuple(sorted(demographics.items()))

            if demo_pattern in demographic_patterns:
                demographic_duplicates += 1
                existing_patient = demographic_patterns[demo_pattern]
                duplicate_details.append(
                    f"Demographic duplicate: patients '{patient_id}' and '{existing_patient}' have identical demographics"
                )
            else:
                demographic_patterns[demo_pattern] = patient_id

    # 2. Check for time series duplicates (patients with identical time series patterns)
    timeseries_duplicates = 0
    timeseries_patterns = {}

    for patient_id, data in patient_data.items():
        time_series = data.get("time_series", {})
        if isinstance(time_series, dict):
            ts_pattern_dict = {}
            for field_name, ts_data in time_series.items():
                if isinstance(ts_data, list):
                    ts_pattern_dict[field_name] = tuple(
                        tuple(entry) if isinstance(entry, list) else entry
                        for entry in ts_data
                    )
                else:
                    ts_pattern_dict[field_name] = ts_data

            ts_pattern = tuple(sorted(ts_pattern_dict.items()))

            if ts_pattern in timeseries_patterns:
                timeseries_duplicates += 1
                existing_patient = timeseries_patterns[ts_pattern]
                duplicate_details.append(
                    f"Time series duplicate: patients '{patient_id}' and '{existing_patient}' have identical time series patterns"
                )
            else:
                timeseries_patterns[ts_pattern] = patient_id

    total_duplicates = demographic_duplicates + timeseries_duplicates
    unique_patients = total_patients - total_duplicates
    consistency_ratio = unique_patients / total_patients if total_patients > 0 else 1.0

    if total_duplicates == 0:
        return (
            1.0,
            f"No duplicate patients found across {total_patients} patients. All patients have unique demographic and time series patterns.",
        )
    else:
        if len(duplicate_details) > 10:
            shown_details = duplicate_details[:10]
            details_str = (
                "; ".join(shown_details)
                + f"; ... and {len(duplicate_details) - 10} more duplicates"
            )
        else:
            details_str = "; ".join(duplicate_details)

        return (
            consistency_ratio,
            f"Found {total_duplicates} duplicate patients out of {total_patients} ({(total_duplicates / total_patients):.1%}). "
            f"Duplicates include {demographic_duplicates} demographic duplicates and {timeseries_duplicates} time series duplicates. "
            f"Details: {details_str}. Overall consistency: {unique_patients}/{total_patients} unique patients ({consistency_ratio:.3f}).",
        )


def run_all_checks_time_series(patient_data, use_case_config, selected_features=None):
    """
    Execute all quantitative quality checks for time series datasets (UC3).

    Args:
        patient_data: Dictionary containing processed patient data
        use_case_config: Configuration dictionary for UC3
        selected_features: Dictionary of selected demographic features for analysis

    Returns:
        Dictionary of check results in format: {metric_name: (score, explanation)}
    """
    results = {}

    results["population_representativity"] = (
        check_population_representativity_time_series(patient_data, selected_features)
    )
    results["metadata_granularity"] = check_metadata_granularity_time_series(
        patient_data
    )

    expected_ranges = {"age": (0, 120)}

    if (
        use_case_config.get("other_column")
        and use_case_config["other_column"] != "None"
    ):
        other_field = use_case_config["other_column"]
        field_ranges = {
            "gender": ["Male", "Female"],
            "diabeticStatus": [0, 1, 2],
            "weight": (0, 300),
            "hospitalID": None,
        }

        if other_field in field_ranges and field_ranges[other_field] is not None:
            expected_ranges[other_field] = field_ranges[other_field]

    results["accuracy"] = check_accuracy_time_series(patient_data, expected_ranges)
    results["coherence"] = check_coherence_time_series(
        patient_data, use_case_config.get("column_types", {})
    )
    results["semantic_coherence"] = check_semantic_coherence_time_series(patient_data)
    results["completeness"] = check_completeness_time_series(patient_data)
    results["relational_consistency"] = check_relational_consistency_time_series(
        patient_data
    )

    return results
