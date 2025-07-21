import os
from pathlib import Path

import SimpleITK as sitk
import numpy as np
import pydicom


def find_dicom_files(directory):
    """
    Find all DICOM files in directory structure.

    :param directory: Root directory to search
    :type directory: str
    :return: Dictionary mapping patient IDs to DICOM file paths
    :rtype: dict
    """
    patient_dict = {}

    for root, dirs, files in os.walk(directory):
        dcm_files = []
        for file in files:
            file_path = os.path.join(root, file)

            if file.lower().endswith(".dcm") or file.lower().endswith(".dicom"):
                dcm_files.append(file_path)
            else:
                try:
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                    if hasattr(ds, "SOPClassUID"):
                        dcm_files.append(file_path)
                except:
                    continue

        if dcm_files:
            path_parts = Path(root).parts
            patient_id = None

            for part in path_parts:
                if "LUNG1-" in part:
                    patient_id = part
                    break

            if not patient_id:
                patient_id = Path(root).name

            if patient_id not in patient_dict:
                patient_dict[patient_id] = []
            patient_dict[patient_id].extend(dcm_files)

    return patient_dict


def read_dicom_series(dicom_files):
    """
    Read and sort DICOM files for a series.

    :param dicom_files: List of DICOM file paths
    :type dicom_files: list
    :return: Sorted list of DICOM datasets
    :rtype: list
    """
    dicom_slices = []

    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(file_path, force=True)

            if hasattr(ds, "Modality"):
                if ds.Modality in [
                    "RTSTRUCT",
                    "RTPLAN",
                    "RTDOSE",
                    "RTIMAGE",
                    "REG",
                    "SEG",
                ]:
                    continue

            if (
                hasattr(ds, "pixel_array")
                and hasattr(ds, "Rows")
                and hasattr(ds, "Columns")
            ):
                dicom_slices.append(ds)

        except Exception:
            continue

    if not dicom_slices:
        return []

    try:
        dicom_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except (AttributeError, TypeError, IndexError):
        try:
            dicom_slices.sort(key=lambda x: int(x.InstanceNumber))
        except (AttributeError, TypeError):
            try:
                dicom_slices.sort(key=lambda x: float(x.SliceLocation))
            except (AttributeError, TypeError):
                pass

    return dicom_slices


def dicom_to_sitk_image(dicom_slices):
    """
    Convert DICOM series to SimpleITK image.

    :param dicom_slices: Sorted list of DICOM datasets
    :type dicom_slices: list
    :return: SimpleITK image
    :rtype: sitk.Image
    """
    if not dicom_slices:
        raise ValueError("No DICOM slices provided")

    first_slice = dicom_slices[0]

    pixel_arrays = []
    for ds in dicom_slices:
        pixel_array = ds.pixel_array.astype(np.float64)

        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            pixel_array = pixel_array * slope + intercept

        pixel_arrays.append(pixel_array)

    volume_array = np.stack(pixel_arrays, axis=0)

    if volume_array.min() >= 0 and volume_array.max() <= 65535:
        data_type = np.uint16
    else:
        data_type = np.int16

    sitk_image = sitk.GetImageFromArray(volume_array.astype(data_type))

    try:
        pixel_spacing = first_slice.PixelSpacing
        x_spacing = float(pixel_spacing[0])
        y_spacing = float(pixel_spacing[1])

        if len(dicom_slices) > 1:
            try:
                pos1 = dicom_slices[0].ImagePositionPatient[2]
                pos2 = dicom_slices[1].ImagePositionPatient[2]
                z_spacing = abs(float(pos2) - float(pos1))
            except (AttributeError, TypeError, IndexError):
                z_spacing = float(getattr(first_slice, "SliceThickness", 1.0))
        else:
            z_spacing = float(getattr(first_slice, "SliceThickness", 1.0))

        sitk_image.SetSpacing((x_spacing, y_spacing, z_spacing))

    except (AttributeError, TypeError):
        sitk_image.SetSpacing((1.0, 1.0, 1.0))

    try:
        origin = first_slice.ImagePositionPatient
        sitk_image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))
    except (AttributeError, TypeError):
        sitk_image.SetOrigin((0.0, 0.0, 0.0))

    return sitk_image


def convert_dcm_to_nrrd(dcm_directory, output_directory="assets/converted_nrrds"):
    """
    Convert DCM files to NRRD format using minimal dependencies.

    :param dcm_directory: Path to directory containing DCM files
    :type dcm_directory: str
    :param output_directory: Path to output directory for NRRD files
    :type output_directory: str
    :return: Success status, message, and total DCM slices processed
    :rtype: tuple
    """
    try:
        os.makedirs(output_directory, exist_ok=True)

        patient_dict = find_dicom_files(dcm_directory)

        if not patient_dict:
            return False, "No DICOM files found in the specified directory", 0

        converted_count = 0
        failed_count = 0
        total_dcm_slices = 0

        for patient_id, dicom_files in patient_dict.items():
            try:
                dicom_slices = read_dicom_series(dicom_files)

                if not dicom_slices:
                    failed_count += 1
                    continue

                total_dcm_slices += len(dicom_slices)

                sitk_image = dicom_to_sitk_image(dicom_slices)

                patient_output_dir = os.path.join(output_directory, patient_id)
                os.makedirs(patient_output_dir, exist_ok=True)

                output_path = os.path.join(patient_output_dir, "image.nrrd")
                sitk.WriteImage(sitk_image, output_path)

                converted_count += 1

            except Exception:
                failed_count += 1
                continue

        if converted_count > 0:
            if failed_count > 0:
                message = f"({failed_count} failed)"
            else:
                message = ""
            return True, message, total_dcm_slices
        else:
            return (
                False,
                f"Failed to convert any patients. {failed_count} failures total.",
                0,
            )

    except Exception as e:
        return False, f"Error during DICOM to NRRD conversion: {str(e)}", 0
