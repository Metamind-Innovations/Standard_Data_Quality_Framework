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
    Check population representativity for image datasets using clinical metadata.

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

        total_samples = len(available_clinical.dropna(subset=['Histology']))
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
        return score, f"Found {num_classes} histology types across {len(available_clinical)} patients with images. Distribution: {details_str}"

    else:
        # Fallback to patient image count distribution
        patient_counts = Counter()
        for path in image_paths:
            patient_id = extract_patient_id_from_path(path)
            patient_counts[patient_id] += 1

        num_patients = len(patient_counts)
        if num_patients == 0:
            return 0, "No patients found"

        image_counts = list(patient_counts.values())
        avg_images = np.mean(image_counts)
        std_dev = np.std(image_counts)
        cv = std_dev / avg_images if avg_images > 0 else 0

        # Score based on coefficient of variation (lower CV = better balance)
        score = max(0, 1 - cv)
        return score, f"Found {num_patients} patients with {len(image_paths)} total images. Average {avg_images:.1f} images per patient (CV: {cv:.2f})"


def check_metadata_granularity_images(image_paths, metadata=None):
    """
    Check metadata availability for image datasets.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame or path to CSV
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    # Check embedded DICOM metadata in images
    total_images = len(image_paths)
    images_with_metadata = 0

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            metadata_keys = image.GetMetaDataKeys()
            # Consider meaningful metadata (more than just basic image info)
            if len(metadata_keys) > 5:
                images_with_metadata += 1
        except:
            continue

    embedded_ratio = images_with_metadata / total_images

    # Load clinical metadata if available
    if metadata is None or isinstance(metadata, str):
        clinical_data = load_clinical_metadata(metadata)
    else:
        clinical_data = metadata

    if clinical_data is not None:
        # Check clinical metadata coverage
        image_patient_ids = set()
        for path in image_paths:
            patient_id = extract_patient_id_from_path(path)
            image_patient_ids.add(patient_id)

        patients_with_clinical = clinical_data[clinical_data['PatientID'].isin(image_patient_ids)]
        clinical_coverage = len(patients_with_clinical) / len(image_patient_ids) if len(image_patient_ids) > 0 else 0

        # Check completeness of clinical data
        clinical_completeness = 0
        if len(patients_with_clinical) > 0:
            non_na_counts = patients_with_clinical.count()
            total_fields = len(patients_with_clinical.columns)
            clinical_completeness = non_na_counts.mean() / len(patients_with_clinical)

        combined_score = (embedded_ratio + clinical_coverage + clinical_completeness) / 3

        return combined_score, f"Embedded metadata: {images_with_metadata}/{total_images} images ({embedded_ratio:.1%}). Clinical metadata: {len(patients_with_clinical)}/{len(image_patient_ids)} patients ({clinical_coverage:.1%}). Clinical completeness: {clinical_completeness:.1%}"
    else:
        return embedded_ratio, f"Embedded metadata only: {images_with_metadata}/{total_images} images ({embedded_ratio:.1%}). No clinical metadata found"


def check_accuracy_images(image_paths, metadata=None):
    """
    Check accuracy of image data including clinical data validation.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame or path to CSV
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

            # HU value range check
            if expected_hu_range[0] <= array.min() and array.max() <= expected_hu_range[1]:
                passed_checks += 1
            else:
                issues.append(f"{Path(path).name}: HU values outside expected range")

            # Pixel spacing check
            spacing_check = all(expected_spacing_range[0] <= s <= expected_spacing_range[1] for s in spacing[:2])
            if spacing_check:
                passed_checks += 1
            else:
                issues.append(f"{Path(path).name}: Pixel spacing outside expected range")

            # Slice thickness check
            if expected_spacing_range[0] <= spacing[2] <= expected_spacing_range[1]:
                passed_checks += 1
            else:
                issues.append(f"{Path(path).name}: Slice thickness outside expected range")

            # Dimension check
            dim_check = all(expected_dim_range[0] <= d <= expected_dim_range[1] for d in size)
            if dim_check:
                passed_checks += 1
            else:
                issues.append(f"{Path(path).name}: Image dimensions outside expected range")

        except Exception:
            issues.append(f"{Path(path).name}: Error reading image")

    # Clinical data accuracy checks
    if metadata is None or isinstance(metadata, str):
        clinical_data = load_clinical_metadata(metadata)
    else:
        clinical_data = metadata

    if clinical_data is not None:
        clinical_checks = 0
        clinical_passed = 0

        # Age range check
        if 'age' in clinical_data.columns:
            age_data = pd.to_numeric(clinical_data['age'], errors='coerce').dropna()
            clinical_checks += len(age_data)
            clinical_passed += ((age_data >= 0) & (age_data <= 120)).sum()

        # Survival time check
        if 'Survival.time' in clinical_data.columns:
            survival_data = pd.to_numeric(clinical_data['Survival.time'], errors='coerce').dropna()
            clinical_checks += len(survival_data)
            clinical_passed += ((survival_data >= 0) & (survival_data <= 10000)).sum()

        # Stage consistency check
        stage_columns = ['clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage']
        for col in stage_columns:
            if col in clinical_data.columns:
                stage_data = pd.to_numeric(clinical_data[col], errors='coerce').dropna()
                clinical_checks += len(stage_data)
                if 'T.Stage' in col:
                    clinical_passed += ((stage_data >= 1) & (stage_data <= 5)).sum()
                elif 'N.Stage' in col:
                    clinical_passed += ((stage_data >= 0) & (stage_data <= 4)).sum()
                elif 'M.Stage' in col:
                    clinical_passed += ((stage_data >= 0) & (stage_data <= 3)).sum()

        total_checks += clinical_checks
        passed_checks += clinical_passed

    accuracy_ratio = passed_checks / total_checks if total_checks > 0 else 0

    if issues:
        issues_summary = "; ".join(issues[:3])
        if len(issues) > 3:
            issues_summary += f" and {len(issues) - 3} more issues"
        return accuracy_ratio, f"Accuracy checks: {passed_checks}/{total_checks} passed ({accuracy_ratio:.1%}). Issues: {issues_summary}"
    else:
        return accuracy_ratio, f"All accuracy checks passed: {passed_checks}/{total_checks} ({accuracy_ratio:.1%})"


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
        'size': [],
        'pixel_type': []
    }

    for path in image_paths:
        try:
            image = sitk.ReadImage(path)
            # Round spacing to avoid minor floating point differences
            spacing = tuple(round(s, 2) for s in image.GetSpacing())
            properties['spacing'].append(spacing)
            properties['direction'].append(image.GetDirection())
            properties['size'].append(image.GetSize())
            properties['pixel_type'].append(image.GetPixelIDTypeAsString())
        except:
            continue

    coherence_scores = {}

    # Spacing coherence
    unique_spacings = len(set(properties['spacing']))
    coherence_scores['spacing'] = 1.0 if unique_spacings <= 2 else max(0.2, 2.0 / unique_spacings)

    # Direction coherence
    unique_directions = len(set(properties['direction']))
    coherence_scores['direction'] = 1.0 if unique_directions == 1 else max(0.2, 1.0 / unique_directions)

    # Size coherence (some variation is acceptable for medical images)
    unique_sizes = len(set(properties['size']))
    coherence_scores['size'] = 1.0 if unique_sizes <= 3 else max(0.2, 3.0 / unique_sizes)

    # Pixel type coherence
    unique_types = len(set(properties['pixel_type']))
    coherence_scores['pixel_type'] = 1.0 if unique_types == 1 else max(0.2, 1.0 / unique_types)

    overall_coherence = np.mean(list(coherence_scores.values()))

    issues = []
    if unique_spacings > 2:
        issues.append(f"{unique_spacings} different spacings")
    if unique_directions > 1:
        issues.append(f"{unique_directions} different orientations")
    if unique_sizes > 3:
        issues.append(f"{unique_sizes} different image sizes")
    if unique_types > 1:
        issues.append(f"{unique_types} different pixel types")

    if issues:
        return overall_coherence, f"Coherence score: {overall_coherence:.2f}. Inconsistencies: {', '.join(issues)}"
    else:
        return overall_coherence, f"Good coherence across all image properties (score: {overall_coherence:.2f})"


def check_semantic_coherence_images(image_paths):
    """
    Check for semantic coherence including duplicate patient IDs and naming consistency.

    :param image_paths: List of paths to NRRD image files
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    patient_series = []
    naming_patterns = []

    for path in image_paths:
        patient_id = extract_patient_id_from_path(path)
        series_type = Path(path).stem
        patient_series.append((patient_id, series_type))

        # Check naming pattern consistency
        filename = Path(path).name
        if 'image' in filename.lower():
            naming_patterns.append('image')
        elif 'volume' in filename.lower():
            naming_patterns.append('volume')
        else:
            naming_patterns.append('other')

    # Check for duplicate patient-series combinations
    total_entries = len(patient_series)
    unique_entries = len(set(patient_series))
    duplicates = total_entries - unique_entries

    # Check naming consistency
    pattern_counts = Counter(naming_patterns)
    dominant_pattern_count = max(pattern_counts.values())
    naming_consistency = dominant_pattern_count / len(naming_patterns)

    # Combined score
    duplication_score = unique_entries / total_entries if total_entries > 0 else 1.0
    overall_score = (duplication_score + naming_consistency) / 2

    issues = []
    if duplicates > 0:
        duplicate_counts = Counter(patient_series)
        duplicate_details = [(k, v) for k, v in duplicate_counts.items() if v > 1]
        issues.append(f"{duplicates} duplicate patient-series combinations")

    if naming_consistency < 0.8:
        issues.append(f"Inconsistent naming patterns: {dict(pattern_counts)}")

    if issues:
        return overall_score, f"Semantic coherence issues found: {'; '.join(issues)}. Score: {overall_score:.2f}"
    else:
        return overall_score, f"Good semantic coherence. All {total_entries} entries unique with consistent naming (score: {overall_score:.2f})"


def check_completeness_images(image_paths, metadata=None):
    """
    Check completeness of image volumes and associated clinical data.

    :param image_paths: List of paths to NRRD image files
    :param metadata: Optional metadata DataFrame or path to CSV
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    total_volumes = len(image_paths)
    complete_volumes = 0
    issues = []

    # Check image completeness
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

            # Check for empty slices
            slice_sums = np.sum(array, axis=(1, 2))
            empty_slices = np.sum(slice_sums == 0)
            if empty_slices > num_slices * 0.2:
                issues.append(f"{Path(path).name}: {empty_slices}/{num_slices} empty slices")
                continue

            # Check for reasonable HU value distribution
            non_air_voxels = array[array > -900]
            if len(non_air_voxels) < array.size * 0.1:
                issues.append(f"{Path(path).name}: Too few non-air voxels")
                continue

            complete_volumes += 1

        except Exception:
            issues.append(f"{Path(path).name}: Cannot read file")

    image_completeness = complete_volumes / total_volumes if total_volumes > 0 else 0

    # Check clinical data completeness
    if metadata is None or isinstance(metadata, str):
        clinical_data = load_clinical_metadata(metadata)
    else:
        clinical_data = metadata

    clinical_completeness = 1.0  # Default if no clinical data
    if clinical_data is not None:
        image_patient_ids = set()
        for path in image_paths:
            patient_id = extract_patient_id_from_path(path)
            image_patient_ids.add(patient_id)

        patients_with_clinical = clinical_data[clinical_data['PatientID'].isin(image_patient_ids)]

        if len(patients_with_clinical) > 0:
            # Calculate completeness for key clinical fields
            key_fields = ['age', 'gender', 'Histology', 'Overall.Stage']
            available_fields = [f for f in key_fields if f in clinical_data.columns]

            if available_fields:
                field_completeness = []
                for field in available_fields:
                    non_na_count = patients_with_clinical[field].notna().sum()
                    field_completeness.append(non_na_count / len(patients_with_clinical))
                clinical_completeness = np.mean(field_completeness)

    overall_completeness = (image_completeness + clinical_completeness) / 2

    if issues:
        issues_summary = "; ".join(issues[:3])
        if len(issues) > 3:
            issues_summary += f" and {len(issues) - 3} more issues"
        return overall_completeness, f"Image completeness: {complete_volumes}/{total_volumes} ({image_completeness:.1%}). Clinical completeness: {clinical_completeness:.1%}. Issues: {issues_summary}"
    else:
        return overall_completeness, f"High completeness. Images: {complete_volumes}/{total_volumes} ({image_completeness:.1%}). Clinical: {clinical_completeness:.1%}"


def check_relational_consistency_images(image_paths):
    """
    Check for duplicate images and patient ID consistency.

    :param image_paths: List of paths to NRRD image files
    :return: Tuple of (score, explanation)
    :rtype: tuple
    """
    if not image_paths:
        return 0, "No image files found"

    file_hashes = {}
    duplicates = []
    patient_consistency_issues = []

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
                patient_consistency_issues.append(f"{Path(path).name}: Inconsistent patient ID format")

        except Exception:
            continue

    total_files = len(image_paths)
    unique_files = len(file_hashes)
    duplicate_count = total_files - unique_files

    # Calculate consistency scores
    duplication_score = unique_files / total_files if total_files > 0 else 1.0
    consistency_score = (total_files - len(patient_consistency_issues)) / total_files if total_files > 0 else 1.0

    overall_consistency = (duplication_score + consistency_score) / 2

    issues = []
    if duplicate_count > 0:
        dup_examples = [f"{Path(d[0]).name} = {Path(d[1]).name}" for d in duplicates[:2]]
        examples_str = ", ".join(dup_examples)
        if len(duplicates) > 2:
            examples_str += f" and {len(duplicates) - 2} more"
        issues.append(f"{duplicate_count} duplicate images: {examples_str}")

    if patient_consistency_issues:
        issues.append(f"{len(patient_consistency_issues)} patient ID format issues")

    if issues:
        return overall_consistency, f"Relational consistency issues: {'; '.join(issues)}. Score: {overall_consistency:.2f}"
    else:
        return overall_consistency, f"Good relational consistency. All {total_files} images unique with consistent IDs (score: {overall_consistency:.2f})"


def run_all_checks_images(image_paths, metadata=None):
    """
    Run all quantitative checks for image datasets with enhanced clinical integration.

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
