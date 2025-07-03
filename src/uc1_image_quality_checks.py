from collections import Counter
from pathlib import Path

import hashlib
import SimpleITK as sitk
import numpy as np


def check_population_representativity_images(image_paths, metadata=None):
    """
    Check population representativity for image datasets.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    if metadata is not None and 'Histology' in metadata.columns:
        histology_counts = metadata['Histology'].value_counts()
        num_classes = len(histology_counts)

        if num_classes <= 1:
            return 0, "Only one histology type found"

        total_samples = len(metadata)
        ideal_proportion = 1.0 / num_classes

        proportions = histology_counts / total_samples
        deviations = abs(proportions - ideal_proportion)
        max_deviation = deviations.max()

        score = 1.0 - (max_deviation / (1.0 - ideal_proportion))
        score = max(0, min(1, score))

        class_details = []
        for class_name, count in histology_counts.items():
            proportion = count / total_samples
            class_details.append(f"'{class_name}': {count} samples ({proportion:.1%})")

        details_str = ", ".join(class_details)
        return score, f"Found {num_classes} histology types. Distribution: {details_str}"
    else:
        patient_counts = Counter()
        for path in image_paths:
            patient_id = Path(path).parent.name.split('_')[0]
            patient_counts[patient_id] += 1

        num_patients = len(patient_counts)
        if num_patients == 0:
            return 0, "No patients found"

        avg_images_per_patient = sum(patient_counts.values()) / num_patients
        std_dev = np.std(list(patient_counts.values()))
        cv = std_dev / avg_images_per_patient if avg_images_per_patient > 0 else 0

        score = max(0, 1 - cv)
        return score, f"Found {num_patients} patients with average {avg_images_per_patient:.1f} images each (CV: {cv:.2f})"


def check_metadata_granularity_images(image_paths, metadata=None):
    """
    Check metadata availability for image datasets.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    total_images = len(image_paths)
    images_with_metadata = 0

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            metadata_keys = image.GetMetaDataKeys()
            if len(metadata_keys) > 0:
                images_with_metadata += 1
        except:
            continue

    if metadata is not None:
        patient_ids = set()
        for path in image_paths:
            patient_id = Path(path).parent.name.split('_')[0]
            patient_ids.add(patient_id)

        patients_in_metadata = len(metadata) if 'PatientID' in metadata.columns else 0
        metadata_coverage = patients_in_metadata / len(patient_ids) if len(patient_ids) > 0 else 0

        combined_score = (images_with_metadata / total_images + metadata_coverage) / 2
        return combined_score, f"Images with embedded metadata: {images_with_metadata}/{total_images}. Clinical metadata coverage: {patients_in_metadata}/{len(patient_ids)} patients"
    else:
        ratio = images_with_metadata / total_images
        return ratio, f"Images with embedded metadata: {images_with_metadata}/{total_images} ({ratio:.2%})"


def check_accuracy_images(image_paths):
    """
    Check accuracy of image data (pixel value ranges, spacing, dimensions).

    :param image_paths: List of paths to NRRD image files
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    total_checks = 0
    passed_checks = 0
    issues = []

    expected_hu_range = (-1024, 3071)
    expected_spacing_range = (0.5, 5.0)
    expected_dim_range = (64, 1024)

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(image)
            spacing = image.GetSpacing()
            size = image.GetSize()

            total_checks += 4

            if expected_hu_range[0] <= array.min() and array.max() <= expected_hu_range[1]:
                passed_checks += 1
            else:
                issues.append(f"{Path(path).name}: HU values outside expected range")

            spacing_check = all(expected_spacing_range[0] <= s <= expected_spacing_range[1] for s in spacing[:2])
            if spacing_check:
                passed_checks += 1
            else:
                issues.append(f"{Path(path).name}: Pixel spacing outside expected range")

            if expected_spacing_range[0] <= spacing[2] <= expected_spacing_range[1]:
                passed_checks += 1
            else:
                issues.append(f"{Path(path).name}: Slice thickness outside expected range")

            dim_check = all(expected_dim_range[0] <= d <= expected_dim_range[1] for d in size)
            if dim_check:
                passed_checks += 1
            else:
                issues.append(f"{Path(path).name}: Image dimensions outside expected range")

        except Exception as e:
            issues.append(f"{Path(path).name}: Error reading image")

    accuracy_ratio = passed_checks / total_checks if total_checks > 0 else 0

    if issues:
        issues_summary = "; ".join(issues[:3])
        if len(issues) > 3:
            issues_summary += f" and {len(issues) - 3} more issues"
        return accuracy_ratio, f"Accuracy checks: {passed_checks}/{total_checks} passed. Issues: {issues_summary}"
    else:
        return accuracy_ratio, f"All accuracy checks passed: {passed_checks}/{total_checks}"


def check_coherence_images(image_paths):
    """
    Check coherence of image properties across the dataset.

    :param image_paths: List of paths to NRRD image files
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    properties = {
        'spacing': [],
        'direction': [],
        'origin': [],
        'size': [],
        'pixel_type': []
    }

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            properties['spacing'].append(image.GetSpacing())
            properties['direction'].append(image.GetDirection())
            properties['origin'].append(tuple(round(x, 2) for x in image.GetOrigin()))
            properties['size'].append(image.GetSize())
            properties['pixel_type'].append(image.GetPixelIDTypeAsString())
        except:
            continue

    coherence_scores = {}

    unique_spacings = len(set(properties['spacing']))
    coherence_scores['spacing'] = 1.0 if unique_spacings == 1 else 1.0 / unique_spacings

    unique_directions = len(set(properties['direction']))
    coherence_scores['direction'] = 1.0 if unique_directions == 1 else 1.0 / unique_directions

    unique_sizes = len(set(properties['size']))
    coherence_scores['size'] = 1.0 if unique_sizes <= 5 else 5.0 / unique_sizes

    unique_types = len(set(properties['pixel_type']))
    coherence_scores['pixel_type'] = 1.0 if unique_types == 1 else 1.0 / unique_types

    overall_coherence = np.mean(list(coherence_scores.values()))

    issues = []
    if unique_spacings > 1:
        issues.append(f"{unique_spacings} different spacings")
    if unique_directions > 1:
        issues.append(f"{unique_directions} different orientations")
    if unique_sizes > 5:
        issues.append(f"{unique_sizes} different image sizes")
    if unique_types > 1:
        issues.append(f"{unique_types} different pixel types")

    if issues:
        return overall_coherence, f"Coherence score: {overall_coherence:.2f}. Inconsistencies found: {', '.join(issues)}"
    else:
        return overall_coherence, "All images have consistent properties"


def check_semantic_coherence_images(image_paths):
    """
    Check for duplicate patient IDs or series in image dataset.

    :param image_paths: List of paths to NRRD image files
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    patient_series = []
    for path in image_paths:
        parts = Path(path).parent.name.split('_')
        patient_id = parts[0] if parts else "unknown"
        series_type = Path(path).stem
        patient_series.append((patient_id, series_type))

    total_entries = len(patient_series)
    unique_entries = len(set(patient_series))
    duplicates = total_entries - unique_entries

    score = unique_entries / total_entries if total_entries > 0 else 1.0

    if duplicates > 0:
        duplicate_counts = Counter(patient_series)
        duplicate_details = [(k, v) for k, v in duplicate_counts.items() if v > 1]
        details_str = ", ".join([f"{k}: {v} copies" for k, v in duplicate_details[:3]])
        if len(duplicate_details) > 3:
            details_str += f" and {len(duplicate_details) - 3} more"
        return score, f"Found {duplicates} duplicate patient-series combinations. Examples: {details_str}"
    else:
        return score, f"No duplicate patient-series combinations found. All {total_entries} entries are unique"


def check_completeness_images(image_paths):
    """
    Check completeness of image volumes (missing slices, corrupted files).

    :param image_paths: List of paths to NRRD image files
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    total_volumes = len(image_paths)
    complete_volumes = 0
    issues = []

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            array = sitk.GetArrayFromImage(image)

            if len(array.shape) != 3:
                issues.append(f"{Path(path).name}: Not a 3D volume")
                continue

            num_slices = array.shape[0]
            if num_slices < 10:
                issues.append(f"{Path(path).name}: Only {num_slices} slices")
                continue

            slice_sums = np.sum(array, axis=(1, 2))
            empty_slices = np.sum(slice_sums == 0)
            if empty_slices > num_slices * 0.1:
                issues.append(f"{Path(path).name}: {empty_slices}/{num_slices} empty slices")
                continue

            complete_volumes += 1

        except Exception as e:
            issues.append(f"{Path(path).name}: Cannot read file")

    completeness_ratio = complete_volumes / total_volumes if total_volumes > 0 else 0

    if issues:
        issues_summary = "; ".join(issues[:3])
        if len(issues) > 3:
            issues_summary += f" and {len(issues) - 3} more issues"
        return completeness_ratio, f"Complete volumes: {complete_volumes}/{total_volumes}. Issues: {issues_summary}"
    else:
        return completeness_ratio, f"All {total_volumes} volumes are complete with no missing data"


def check_relational_consistency_images(image_paths):
    """
    Check for duplicate images based on file hash or content similarity.

    :param image_paths: List of paths to NRRD image files
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    file_hashes = {}
    duplicates = []

    for path in image_paths:
        try:
            with open(path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in file_hashes:
                duplicates.append((path, file_hashes[file_hash]))
            else:
                file_hashes[file_hash] = path
        except:
            continue

    total_files = len(image_paths)
    unique_files = len(file_hashes)
    duplicate_count = total_files - unique_files

    consistency_ratio = unique_files / total_files if total_files > 0 else 1.0

    if duplicate_count > 0:
        dup_examples = [f"{Path(d[0]).name} = {Path(d[1]).name}" for d in duplicates[:3]]
        examples_str = ", ".join(dup_examples)
        if len(duplicates) > 3:
            examples_str += f" and {len(duplicates) - 3} more"
        return consistency_ratio, f"Found {duplicate_count} duplicate images. Examples: {examples_str}"
    else:
        return consistency_ratio, f"No duplicate images found. All {total_files} images are unique"


def run_all_checks_images(image_paths, metadata=None):
    """
    Run all quantitative checks for image datasets.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame
    :return: Dictionary of check results
    :rtype: dict
    """
    results = {}

    results["population_representativity"] = check_population_representativity_images(image_paths, metadata)
    results["metadata_granularity"] = check_metadata_granularity_images(image_paths, metadata)
    results["accuracy"] = check_accuracy_images(image_paths)
    results["coherence"] = check_coherence_images(image_paths)
    results["semantic_coherence"] = check_semantic_coherence_images(image_paths)
    results["completeness"] = check_completeness_images(image_paths)
    results["relational_consistency"] = check_relational_consistency_images(image_paths)

    return results
