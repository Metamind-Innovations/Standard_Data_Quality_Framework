import hashlib
import os
from collections import Counter
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pandas as pd


def extract_patient_id_from_path(path):
    """
    Extract patient ID from image file path.

    :param path: Path to image file
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


def load_clinical_metadata(metadata_path=None):
    """
    Load clinical metadata from CSV file.

    :param metadata_path: Path to clinical CSV file
    :return: Clinical metadata DataFrame
    :rtype: pd.DataFrame or None
    """
    if metadata_path is None:
        # Try default locations
        default_paths = [
            "assets/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv",
            "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
        ]

        for path in default_paths:
            if os.path.exists(path):
                metadata_path = path
                break

    if metadata_path and os.path.exists(metadata_path):
        try:
            return pd.read_csv(metadata_path)
        except Exception as e:
            print(f"Error loading clinical metadata: {e}")
            return None

    return None


def check_population_representativity_images(image_paths, metadata=None):
    """
    Check population representativity using minority/majority class ratio as per SDQF guidelines.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame or path to CSV
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    # Load clinical metadata if not provided
    if metadata is None or isinstance(metadata, str):
        clinical_data = load_clinical_metadata(metadata)
    else:
        clinical_data = metadata

    if clinical_data is not None and 'Histology' in clinical_data.columns:
        # Extract patient IDs from image paths
        image_patient_ids = set()
        for path in image_paths:
            patient_id = extract_patient_id_from_path(path)
            image_patient_ids.add(patient_id)

        # Filter clinical data to only include patients with images
        available_clinical = clinical_data[clinical_data['PatientID'].isin(image_patient_ids)]

        if len(available_clinical) == 0:
            return 0, "No clinical data found for patients with images"

        # Check histology distribution
        histology_counts = available_clinical['Histology'].dropna().value_counts()
        num_classes = len(histology_counts)

        if num_classes <= 1:
            return 0, "Only one histology type found in available data"

        minority_class_count = histology_counts.min()
        majority_class_count = histology_counts.max()

        ratio = minority_class_count / majority_class_count

        if ratio <= 0.2:
            score_rating = 1
        elif ratio >= 0.8:
            score_rating = 5
        else:
            score_rating = 1 + (ratio - 0.2) / 0.15
            score_rating = min(5, max(1, score_rating))

        score = (score_rating - 1) / 4

        class_details = []
        for class_name, count in histology_counts.items():
            class_details.append(f"'{class_name}': {count} samples")

        details_str = ", ".join(class_details)
        return score, f"Minority/majority ratio: {ratio:.3f}. Classes: {details_str}. Rating: {score_rating:.1f}/5"

    else:
        # Fallback to patient image count distribution
        patient_counts = Counter()
        for path in image_paths:
            patient_id = extract_patient_id_from_path(path)
            patient_counts[patient_id] += 1

        if len(patient_counts) == 0:
            return 0, "No patients found"

        image_counts = list(patient_counts.values())
        min_images = min(image_counts)
        max_images = max(image_counts)

        ratio = min_images / max_images if max_images > 0 else 0

        if ratio <= 0.2:
            score_rating = 1
        elif ratio >= 0.8:
            score_rating = 5
        else:
            score_rating = 1 + (ratio - 0.2) / 0.15
            score_rating = min(5, max(1, score_rating))

        score = (score_rating - 1) / 4

        return score, f"Patient image distribution - Min/max ratio: {ratio:.3f}. Range: {min_images}-{max_images} images per patient. Rating: {score_rating:.1f}/5"


def check_metadata_granularity_images(image_paths, metadata=None):
    """
    Check metadata granularity based on patients with metadata / total patients ratio.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame or path to CSV
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

    # Load clinical metadata if available
    if metadata is None or isinstance(metadata, str):
        clinical_data = load_clinical_metadata(metadata)
    else:
        clinical_data = metadata

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
        return 0, "No clinical metadata available"


def check_accuracy_images(image_paths, metadata=None):
    """
    Check accuracy focusing on slice dimensions consistency and missing slices.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame or path to CSV
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
    Check coherence focusing on consistent number of channels across images.

    :param image_paths: List of paths to NRRD image files
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
    Check semantic coherence by detecting duplicate images/slices.

    :param image_paths: List of paths to NRRD image files
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


def check_completeness_images(image_paths, metadata=None):
    """
    Check completeness by calculating missing pixels / total pixels ratio.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame or path to CSV
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


def check_relational_consistency_images(image_paths):
    """
    Check relational consistency by detecting duplicate files and patient ID consistency.

    :param image_paths: List of paths to NRRD image files
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    file_hashes = {}
    duplicates = []
    patient_consistency_issues = 0
    total_files = len(image_paths)

    for path in image_paths:
        try:
            # Check for file duplicates
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in file_hashes:
                duplicates.append((path, file_hashes[file_hash]))
            else:
                file_hashes[file_hash] = path

            # Check patient ID consistency in path structure
            patient_id = extract_patient_id_from_path(path)
            if not patient_id.startswith('LUNG1-'):
                patient_consistency_issues += 1

        except Exception:
            continue

    duplicate_count = len(duplicates)
    duplicate_ratio = duplicate_count / total_files
    inconsistency_ratio = patient_consistency_issues / total_files

    combined_issues_ratio = (duplicate_ratio + inconsistency_ratio) / 2

    if combined_issues_ratio <= 0.2:
        score_rating = 5
    elif combined_issues_ratio >= 0.8:
        score_rating = 1
    else:
        score_rating = 5 - (combined_issues_ratio - 0.2) / 0.15
        score_rating = min(5, max(1, score_rating))

    score = (score_rating - 1) / 4

    unique_files = total_files - duplicate_count
    consistent_patient_ids = total_files - patient_consistency_issues

    issues = []
    if duplicate_count > 0:
        issues.append(f"{duplicate_count} duplicate files")
    if patient_consistency_issues > 0:
        issues.append(f"{patient_consistency_issues} patient ID format issues")

    if issues:
        issues_str = ", ".join(issues)
        return score, f"Relational consistency issues: {issues_str}. Unique files: {unique_files}/{total_files}. Consistent patient IDs: {consistent_patient_ids}/{total_files}. Combined issues ratio: {combined_issues_ratio:.3f}. Rating: {score_rating:.1f}/5"
    else:
        return score, f"Good relational consistency. All {total_files} files unique with consistent patient IDs. Rating: {score_rating:.1f}/5"


def run_all_checks_images(image_paths, metadata=None):
    """
    Run all quantitative checks for image datasets following SDQF guidelines.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame or path to CSV
    :return: Dictionary of check results
    :rtype: dict
    """
    results = {}

    results["population_representativity"] = check_population_representativity_images(image_paths, metadata)
    results["metadata_granularity"] = check_metadata_granularity_images(image_paths, metadata)
    results["accuracy"] = check_accuracy_images(image_paths, metadata)
    results["coherence"] = check_coherence_images(image_paths)
    results["semantic_coherence"] = check_semantic_coherence_images(image_paths)
    results["completeness"] = check_completeness_images(image_paths, metadata)
    results["relational_consistency"] = check_relational_consistency_images(image_paths)

    return results
