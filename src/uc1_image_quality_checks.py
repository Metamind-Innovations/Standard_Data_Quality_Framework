import hashlib
import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pandas as pd


def extract_patient_id_from_path(path):
    """
    Extract patient ID from image file path using various naming conventions.

    Utility function that parses file and directory paths to identify patient identifiers,
    supporting different naming patterns commonly found in medical imaging datasets.

    :param path: Path to image file
    :type path: str
    :return: Patient ID string
    :rtype: str
    """
    path_obj = Path(path)

    if '_' in path_obj.parent.name:
        return path_obj.parent.name.split('_')[0]
    elif 'LUNG1-' in path_obj.parent.name:
        return path_obj.parent.name
    elif 'LUNG1-' in path_obj.name:
        parts = path_obj.name.split('_')
        for part in parts:
            if 'LUNG1-' in part:
                return part.replace('.nrrd', '').replace('.mha', '')

    return path_obj.parent.name


def load_clinical_metadata(metadata_path):
    """
    Load clinical metadata from CSV file with automatic path detection.

    Utility function that loads clinical metadata to support enhanced quality assessments.
    Attempts to locate metadata files in default locations if no path is specified.

    :param metadata_path: Path to clinical CSV file
    :type metadata_path: str or None
    :return: Clinical metadata DataFrame
    :rtype: pd.DataFrame or None
    """
    if metadata_path and os.path.exists(metadata_path):
        try:
            return pd.read_csv(metadata_path)
        except Exception as e:
            print(f"Error loading clinical metadata: {e}")
            return None

    return None


def check_population_representativity_images(image_paths, metadata=None, selected_features=None):
    """
    Assess population representativity through class distribution analysis across multiple features.

    **Population Representativity Check**

    Population representativity refers to the degree to which the data adequately
    represent the population in question. For image data, this is measured by checking
    each class has a similar number of samples using the minority/majority class ratio
    across multiple selected features.

    **SDQF Rating Criteria:**

    * Minority/majority ratio ≤0.2: Rating 1/5 (poor representativity)
    * Minority/majority ratio ≥0.8: Rating 5/5 (excellent representativity)

    The function analyzes distribution across selected features (histology, gender, age, subpopulation).
    If no clinical metadata is provided, returns score of 0 as population representativity
    cannot be assessed without demographic/clinical information.

    :param image_paths: List of paths to NRRD image files
    :type image_paths: list
    :param metadata: Optional metadata DataFrame or path to CSV
    :type metadata: pd.DataFrame, str, or None
    :param selected_features: Dictionary of selected feature columns
    :type selected_features: dict or None
    :return: Tuple of (score, explanation, detailed_results)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found", {}

    if isinstance(metadata, str):
        clinical_data = load_clinical_metadata(metadata)
    elif isinstance(metadata, pd.DataFrame):
        clinical_data = metadata
    else:
        clinical_data = None

    detailed_results = {}
    feature_scores = []
    feature_explanations = []

    if clinical_data is not None and selected_features:
        # Extract patient IDs from image paths
        image_patient_ids = set()
        for path in image_paths:
            patient_id = extract_patient_id_from_path(path)
            image_patient_ids.add(patient_id)

        # Filter clinical data to only include patients with images
        available_clinical = clinical_data[clinical_data['PatientID'].isin(image_patient_ids)]

        if len(available_clinical) == 0:
            return 0, "No clinical data found for patients with images", {}

        # Analyze each selected feature
        for feature_name, column_name in selected_features.items():
            if column_name and column_name in available_clinical.columns:
                feature_score, feature_explanation, feature_details = _analyze_feature_representativity(
                    available_clinical, column_name, feature_name
                )

                feature_scores.append(feature_score)
                feature_explanations.append(f"{feature_name}: {feature_explanation}")
                detailed_results[feature_name] = {
                    'score': feature_score,
                    'explanation': feature_explanation,
                    'details': feature_details if feature_details else {}
                }
            else:
                # Column not found, add with score 0
                feature_scores.append(0)
                feature_explanations.append(f"{feature_name}: Column '{column_name}' not found in clinical data")
                detailed_results[feature_name] = {
                    'score': 0,
                    'explanation': f"Column '{column_name}' not found in clinical data",
                    'details': {}
                }

        if feature_scores:
            # Calculate average score across all selected features
            average_score = sum(feature_scores) / len(feature_scores)

            explanation_parts = feature_explanations[:3]  # Show first 3 in main explanation
            if len(feature_explanations) > 3:
                explanation_parts.append(f"and {len(feature_explanations) - 3} more features")

            combined_explanation = f"Multi-feature representativity analysis. {'; '.join(explanation_parts)}."

            return average_score, combined_explanation, detailed_results
        else:
            return 0, "No valid features found for representativity analysis", {}

    else:
        return 0, "No clinical metadata available for population representativity analysis", {}


def _analyze_feature_representativity(clinical_data, column_name, feature_name):
    """
    Analyze representativity for a single feature.

    :param clinical_data: Clinical metadata DataFrame
    :type clinical_data: pd.DataFrame
    :param column_name: Name of the column to analyze
    :type column_name: str
    :param feature_name: Display name of the feature
    :type feature_name: str
    :return: Tuple of (score, explanation, details)
    :rtype: tuple
    """
    if 'age' in column_name.lower() or feature_name.lower() == 'age':
        return _analyze_age_representativity(clinical_data, column_name, feature_name)
    else:
        return _analyze_categorical_representativity(clinical_data, column_name, feature_name)


def _analyze_categorical_representativity(clinical_data, column_name, feature_name):
    """
    Analyze representativity for categorical features.

    :param clinical_data: Clinical metadata DataFrame
    :type clinical_data: pd.DataFrame
    :param column_name: Name of the column to analyze
    :type column_name: str
    :param feature_name: Display name of the feature
    :type feature_name: str
    :return: Tuple of (score, explanation, details)
    :rtype: tuple
    """
    feature_data = clinical_data[column_name].dropna()

    if len(feature_data) == 0:
        return 0, f"No valid data for {feature_name}", {}

    class_counts = feature_data.value_counts()
    num_classes = len(class_counts)

    if num_classes <= 1:
        return 0, f"Only one {feature_name.lower()} category found", {}

    # Use new balanced distribution scoring
    score = _calculate_representativity_score(class_counts)

    # Calculate ideal and actual proportions for explanation
    ideal_proportion = 1.0 / num_classes
    total_samples = len(feature_data)

    class_details = []
    for class_name, count in class_counts.items():
        actual_proportion = count / total_samples
        class_details.append(f"'{class_name}': {count} ({actual_proportion:.1%})")

    details_str = ", ".join(class_details[:3])
    if len(class_details) > 3:
        details_str += f" and {len(class_details) - 3} more"

    explanation = f"Balance score {score:.3f} (ideal: {ideal_proportion:.1%} each), Distribution: {details_str}"

    details = {
        'balance_score': score,
        'ideal_proportion': ideal_proportion,
        'classes': dict(class_counts),
        'num_classes': num_classes
    }

    return score, explanation, details


def _analyze_age_representativity(clinical_data, column_name, feature_name):
    """
    Analyze representativity for age feature using age groups.

    :param clinical_data: Clinical metadata DataFrame
    :type clinical_data: pd.DataFrame
    :param column_name: Name of the column to analyze
    :type column_name: str
    :param feature_name: Display name of the feature
    :type feature_name: str
    :return: Tuple of (score, explanation, details)
    :rtype: tuple
    """
    age_data = pd.to_numeric(clinical_data[column_name], errors='coerce').dropna()

    if len(age_data) == 0:
        return 0, f"No valid data for {feature_name}", {}

    # Create age groups
    age_bins = [0, 40, 55, 70, 120]
    age_labels = ['<40', '40-54', '55-69', '70+']

    try:
        age_groups = pd.cut(age_data, bins=age_bins, labels=age_labels, include_lowest=True)
        age_counts = age_groups.value_counts()

        # Filter out age groups with 0 people for balance calculation
        non_empty_age_counts = age_counts[age_counts > 0]

        if len(non_empty_age_counts) <= 1:
            return 0, f"Insufficient {feature_name.lower()} group diversity", {}

        # Use new balanced distribution scoring on non-empty groups
        score = _calculate_representativity_score(non_empty_age_counts)

        # Calculate ideal proportion for explanation
        ideal_proportion = 1.0 / len(non_empty_age_counts)

        group_details = []
        for group_name, count in age_counts.items():
            proportion = count / len(age_data)
            group_details.append(f"{group_name}: {count} ({proportion:.1%})")

        details_str = ", ".join(group_details)
        explanation = f"Age balance score {score:.3f} (ideal: {ideal_proportion:.1%} per group), Distribution: {details_str}"

        details = {
            'balance_score': score,
            'ideal_proportion': ideal_proportion,
            'age_groups': dict(age_counts),
            'mean_age': float(age_data.mean()),
            'age_range': [float(age_data.min()), float(age_data.max())]
        }

        return score, explanation, details

    except Exception as e:
        return 0, f"Error analyzing {feature_name.lower()} groups", {}


def _calculate_representativity_score(class_counts):
    """
    Calculate representativity score based on how balanced the class distribution is.
    Perfect balance (all classes equal) = 1.0, maximum imbalance = 0.0

    :param class_counts: Series or dict with class counts
    :type class_counts: pd.Series or dict
    :return: Score between 0 and 1
    :rtype: float
    """
    if isinstance(class_counts, dict):
        counts = list(class_counts.values())
    else:
        counts = class_counts.values

    num_classes = len(counts)
    total_samples = sum(counts)

    if num_classes <= 1 or total_samples == 0:
        return 0.0

    # Calculate ideal proportion (perfect balance)
    ideal_proportion = 1.0 / num_classes

    # Calculate actual proportions
    actual_proportions = [count / total_samples for count in counts]

    # Calculate deviation from ideal balance
    # Sum of absolute deviations from ideal proportion
    total_deviation = sum(abs(prop - ideal_proportion) for prop in actual_proportions)

    # Maximum possible deviation (when one class has everything, others have nothing)
    max_deviation = 2 * (1 - ideal_proportion)

    # Convert to score (1.0 = perfect balance, 0.0 = maximum imbalance)
    if max_deviation == 0:
        return 1.0

    balance_score = 1.0 - (total_deviation / max_deviation)
    return max(0.0, min(1.0, balance_score))


def check_metadata_granularity_images(image_paths, metadata=None):
    """
    Evaluate metadata granularity based on patient coverage and completeness.

    **Metadata Granularity Check**

    Metadata granularity refers to the availability, comprehensiveness and level of detail
    of metadata that help users understand the data being used. For image data, this is
    measured by checking the ratio of patients with metadata to total patients.

    **SDQF Rating Criteria:**

    * Patients with metadata ratio ≤0.2: Rating 1/5 (poor metadata coverage)
    * Patients with metadata ratio ≥0.8: Rating 5/5 (excellent metadata coverage)

    The function counts patients with at least 50% complete metadata fields as having
    adequate metadata granularity.

    :param image_paths: List of paths to NRRD image files
    :type image_paths: list
    :param metadata: Optional metadata DataFrame or path to CSV
    :type metadata: pd.DataFrame, str, or None
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    image_patient_ids = set()
    for path in image_paths:
        patient_id = extract_patient_id_from_path(path)
        image_patient_ids.add(patient_id)

    total_patients = len(image_patient_ids)

    if isinstance(metadata, str):
        clinical_data = load_clinical_metadata(metadata)
    elif isinstance(metadata, pd.DataFrame):
        clinical_data = metadata
    else:
        clinical_data = None

    if clinical_data is not None:
        patients_with_metadata = clinical_data[clinical_data['PatientID'].isin(image_patient_ids)]
        patients_with_complete_metadata = 0

        for _, patient_row in patients_with_metadata.iterrows():
            non_na_fields = patient_row.notna().sum()
            total_fields = len(patient_row)
            if non_na_fields / total_fields >= 0.5:
                patients_with_complete_metadata += 1

        ratio = patients_with_complete_metadata / total_patients if total_patients > 0 else 0

        if ratio <= 0.2:
            score_rating = 1
        elif ratio >= 0.8:
            score_rating = 5
        else:
            score_rating = 1 + (ratio - 0.2) / 0.15
            score_rating = min(5, max(1, score_rating))

        score = (score_rating - 1) / 4

        return score, f"Patients with metadata: {patients_with_complete_metadata}/{total_patients} ({ratio:.1%}). Rating: {score_rating:.1f}/5"
    else:
        return 0, "No clinical metadata available for metadata granularity analysis"


def check_accuracy_images(image_paths):
    """
    Assess data accuracy through slice dimension consistency and completeness validation.

    **Accuracy Check**

    Accuracy refers to the degree to which data correctly describes what it was designed
    to measure (the "real world" entity). For CT scan images, this is measured by checking
    slice dimension consistency and detecting missing slices.

    **SDQF Rating Criteria:**

    * Missing slices/total slices ratio ≤0.2: Rating 5/5 (excellent accuracy)
    * Missing slices/total slices ratio ≥0.8: Rating 1/5 (poor accuracy)
    * Combined with dimension consistency analysis

    The function evaluates both dimensional consistency across images and identifies
    missing or corrupted slices within volumes.

    :param image_paths: List of paths to NRRD image files
    :type image_paths: list
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    dimension_consistency_issues = 0
    missing_slices_total = 0
    total_expected_slices = 0
    total_volumes = 0
    dimension_groups = {}

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            size = image.GetSize()
            slice_dimensions = (size[0], size[1])

            if slice_dimensions not in dimension_groups:
                dimension_groups[slice_dimensions] = 0
            dimension_groups[slice_dimensions] += 1

            array = sitk.GetArrayFromImage(image)
            if len(array.shape) == 3:
                num_slices = array.shape[0]
                total_expected_slices += num_slices

                slice_sums = np.sum(array, axis=(1, 2))
                missing_slices = np.sum(slice_sums == 0)
                missing_slices_total += missing_slices

                total_volumes += 1
        except Exception:
            dimension_consistency_issues += 1
            continue

    if total_volumes == 0:
        return 0, "No valid image volumes found"

    most_common_dimensions = max(dimension_groups, key=dimension_groups.get)
    images_with_different_dimensions = sum(
        count for dims, count in dimension_groups.items() if dims != most_common_dimensions)

    dimension_inconsistency_ratio = images_with_different_dimensions / len(image_paths)
    missing_slices_ratio = missing_slices_total / total_expected_slices if total_expected_slices > 0 else 0

    combined_error_ratio = (dimension_inconsistency_ratio + missing_slices_ratio) / 2

    if combined_error_ratio <= 0.2:
        score_rating = 5
    elif combined_error_ratio >= 0.8:
        score_rating = 1
    else:
        score_rating = 5 - (combined_error_ratio - 0.2) / 0.15
        score_rating = min(5, max(1, score_rating))

    score = (score_rating - 1) / 4

    return score, f"Dimension consistency: {len(image_paths) - images_with_different_dimensions}/{len(image_paths)} images have consistent dimensions. Missing slices: {missing_slices_total}/{total_expected_slices} ({missing_slices_ratio:.1%}). Combined error ratio: {combined_error_ratio:.3f}. Rating: {score_rating:.1f}/5"


def check_coherence_images(image_paths):
    """
    Evaluate data coherence through consistent image channel analysis.

    **Coherence Check**

    Coherence refers to the degree to which aspects relevant for analysability are used
    consistently throughout a dataset or across different datasets or versions of one dataset.
    For image data, this is measured by checking if images all have the same number of
    channels (e.g., grayscale vs RGB).

    **SDQF Rating Criteria:**

    * Images with different channels/total images ratio ≤0.2: Rating 5/5 (excellent coherence)
    * Images with different channels/total images ratio ≥0.8: Rating 1/5 (poor coherence)

    The function analyzes the number of components per pixel across all images to ensure
    consistent channel representation for proper analysis.

    :param image_paths: List of paths to NRRD image files
    :type image_paths: list
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    channel_counts = {}
    total_images = 0
    processing_errors = 0

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)

            if image.GetNumberOfComponentsPerPixel() == 1:
                channels = 1
            else:
                channels = image.GetNumberOfComponentsPerPixel()

            if channels not in channel_counts:
                channel_counts[channels] = 0
            channel_counts[channels] += 1
            total_images += 1

        except Exception:
            processing_errors += 1
            continue

    if total_images == 0:
        return 0, "No valid images found for channel analysis"

    most_common_channels = max(channel_counts, key=channel_counts.get) if channel_counts else 1
    images_with_different_channels = sum(
        count for channels, count in channel_counts.items() if channels != most_common_channels)

    inconsistency_ratio = images_with_different_channels / total_images

    if inconsistency_ratio <= 0.2:
        score_rating = 5
    elif inconsistency_ratio >= 0.8:
        score_rating = 1
    else:
        score_rating = 5 - (inconsistency_ratio - 0.2) / 0.15
        score_rating = min(5, max(1, score_rating))

    score = (score_rating - 1) / 4

    channel_details = []
    for channels, count in channel_counts.items():
        channel_details.append(f"{channels} channel(s): {count} images")

    details_str = ", ".join(channel_details)

    return score, f"Channel consistency: {total_images - images_with_different_channels}/{total_images} images have consistent channels ({most_common_channels} channels). Distribution: {details_str}. Inconsistency ratio: {inconsistency_ratio:.3f}. Rating: {score_rating:.1f}/5"


def check_semantic_coherence_images(image_paths):
    """
    Assess semantic coherence through duplicate image detection and content analysis.

    **Semantic Coherence Check**

    Semantic coherence refers to the degree to which the same value means the same
    throughout a dataset or across different datasets or versions of one dataset,
    improving analysability. For image data, this is measured by detecting duplicate
    images/slices using content-based hashing.

    **SDQF Rating Criteria:**

    * Number of detected duplicates/total images ratio ≤0.2: Rating 5/5 (excellent semantic coherence)
    * Number of detected duplicates/total images ratio ≥0.8: Rating 1/5 (poor semantic coherence)

    The function uses MD5 hashing of image array content to identify exact duplicates
    that could compromise analysis validity.

    :param image_paths: List of paths to NRRD image files
    :type image_paths: list
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    image_hashes = {}
    total_images = len(image_paths)
    duplicate_images = 0
    processing_errors = 0

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(image)

            array_bytes = array.tobytes()
            image_hash = hashlib.md5(array_bytes).hexdigest()

            if image_hash in image_hashes:
                duplicate_images += 1
            else:
                image_hashes[image_hash] = path

        except Exception:
            processing_errors += 1
            continue

    if total_images - processing_errors == 0:
        return 0, "No valid images found for duplicate analysis"

    valid_images = total_images - processing_errors
    duplication_ratio = duplicate_images / valid_images

    if duplication_ratio <= 0.2:
        score_rating = 5
    elif duplication_ratio >= 0.8:
        score_rating = 1
    else:
        score_rating = 5 - (duplication_ratio - 0.2) / 0.15
        score_rating = min(5, max(1, score_rating))

    score = (score_rating - 1) / 4

    unique_images = valid_images - duplicate_images

    return score, f"Duplicate detection: {duplicate_images}/{valid_images} images are duplicates ({duplication_ratio:.1%}). Unique images: {unique_images}. Duplication ratio: {duplication_ratio:.3f}. Rating: {score_rating:.1f}/5"


def check_completeness_images(image_paths):
    """
    Evaluate data completeness through missing pixel analysis across all images.

    **Completeness Check**

    Completeness refers to the degree to which all required information is present in
    a particular dataset. For image data, this is measured by calculating the ratio of
    missing pixels to total pixels across all images in the dataset.

    **SDQF Rating Criteria:**

    * Missing pixels/total pixels ratio ≤0.2: Rating 5/5 (excellent completeness)
    * Missing pixels/total pixels ratio ≥0.8: Rating 1/5 (poor completeness)

    The function treats zero-valued pixels as missing data, which is appropriate for
    medical imaging where zero typically indicates absence of tissue or data.

    :param image_paths: List of paths to NRRD image files
    :type image_paths: list
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    total_pixels = 0
    missing_pixels = 0
    processing_errors = 0

    # Check image completeness
    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(image)

            current_total_pixels = array.size
            total_pixels += current_total_pixels

            current_missing_pixels = np.sum(array == 0)
            missing_pixels += current_missing_pixels

        except Exception:
            processing_errors += 1
            continue

    if total_pixels == 0:
        return 0, "No valid pixels found for completeness analysis"

    missing_ratio = missing_pixels / total_pixels

    if missing_ratio <= 0.2:
        score_rating = 5
    elif missing_ratio >= 0.8:
        score_rating = 1
    else:
        score_rating = 5 - (missing_ratio - 0.2) / 0.15
        score_rating = min(5, max(1, score_rating))

    score = (score_rating - 1) / 4

    return score, f"Pixel completeness: {total_pixels - missing_pixels}/{total_pixels} pixels are non-zero ({(1 - missing_ratio):.1%} complete). Missing pixels ratio: {missing_ratio:.3f}. Rating: {score_rating:.1f}/5"


def run_all_checks_images(image_paths, metadata=None, selected_features=None):
    """
    Execute all quantitative quality checks for image datasets following SDQF guidelines.

    Orchestrates the complete quality assessment workflow for medical imaging datasets,
    running six core SDQF quality dimensions: population representativity, metadata
    granularity, accuracy, coherence, semantic coherence, and completeness.

    Each check returns standardized scores and explanations following SDQF rating criteria,
    enabling comprehensive quality assessment for medical imaging research and clinical applications.

    :param image_paths: List of paths to NRRD image files
    :type image_paths: list
    :param metadata: Optional metadata DataFrame or path to CSV
    :type metadata: pd.DataFrame, str, or None
    :param selected_features: Dictionary of selected feature columns for representativity analysis
    :type selected_features: dict or None
    :return: Dictionary of check results
    :rtype: dict
    """
    results = {}

    results["population_representativity"] = check_population_representativity_images(image_paths, metadata,
                                                                                      selected_features)
    results["metadata_granularity"] = check_metadata_granularity_images(image_paths, metadata)
    results["accuracy"] = check_accuracy_images(image_paths)
    results["coherence"] = check_coherence_images(image_paths)
    results["semantic_coherence"] = check_semantic_coherence_images(image_paths)
    results["completeness"] = check_completeness_images(image_paths)

    return results
