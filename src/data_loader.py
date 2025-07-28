import io
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd


def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    return None


def load_metadata(uploaded_file):
    if uploaded_file is not None:
        try:
            metadata = pd.read_csv(uploaded_file)
            return metadata
        except Exception as e:
            raise Exception(f"Error loading metadata: {str(e)}")
    return None


def load_uc2_zip_data(uploaded_file):
    """
    Loads and processes a UC2 ZIP file containing VCF files and a CSV ground truth file.

    The function extracts the uploaded ZIP file to a temporary directory, searches for VCF and CSV files,
    and processes them as follows:
      - The CSV file containing "groundtruth" or "phenotype" in its filename is loaded as the ground truth DataFrame.
      - Each VCF file is parsed to separate metadata lines (those starting with "##") and the data table (starting from the "#CHROM" header).
      - For each VCF file, the metadata, data as a DataFrame, and the filename are collected.

    Args:
        uploaded_file: A file-like object representing the uploaded ZIP file.

    Returns:
        groundtruth_df (pd.DataFrame or None): DataFrame containing ground truth phenotype data, or None if not found.
        vcf_dataframes (list of pd.DataFrame): List of DataFrames containing VCF data for each VCF file.
        vcf_metadata (list of list of str): List of metadata lines (strings) for each VCF file.
        vcf_filenames (list of str): List of VCF filenames (basename only).

    Raises:
        Exception: If there is an error processing the ZIP file or its contents.
    """
    if uploaded_file is not None:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                vcf_files = []
                csv_files = []

                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith(".vcf") or file.endswith(".vcf.gz"):
                            vcf_files.append(file_path)
                        elif file.endswith(".csv"):
                            csv_files.append(file_path)

                groundtruth_df = None
                if csv_files:
                    groundtruth_file = None
                    for csv_file in csv_files:
                        filename = os.path.basename(csv_file).lower()
                        if "groundtruth" in filename or "phenotype" in filename:
                            groundtruth_file = csv_file
                            break

                    groundtruth_df = pd.read_csv(groundtruth_file, index_col=0)

                vcf_dataframes = []
                vcf_metadata = []
                vcf_filenames = []

                for vcf_file in vcf_files:
                    vcf_filenames.append(os.path.basename(vcf_file))

                    with open(vcf_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()

                    metadata_lines = []
                    data_start_idx = None

                    for i, line in enumerate(lines):
                        line = line.strip()
                        if line.startswith("##"):
                            metadata_lines.append(line)
                        elif line.startswith("#CHROM"):
                            data_start_idx = i
                            break

                    vcf_metadata.append(metadata_lines)

                    if data_start_idx is not None:
                        header_line = lines[data_start_idx].strip().lstrip("#")
                        headers = header_line.split("\t")

                        data_lines = []
                        for line in lines[data_start_idx + 1 :]:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                data_lines.append(line.split("\t"))

                        if data_lines:
                            vcf_df = pd.DataFrame(data_lines, columns=headers)
                            vcf_dataframes.append(vcf_df)
                        else:
                            vcf_df = pd.DataFrame(columns=headers)
                            vcf_dataframes.append(vcf_df)
                    else:
                        vcf_df = pd.DataFrame()
                        vcf_dataframes.append(vcf_df)

                return groundtruth_df, vcf_dataframes, vcf_metadata, vcf_filenames

        except Exception as e:
            raise Exception(f"Error processing UC2 zip file: {str(e)}")

    return None, [], [], []


def load_uc3_zip_data(uploaded_file):
    """
    Loads and processes a UC3 zip file containing JSON files with glucose control data for multiple patients.

    The function extracts all JSON files from the provided zip archive, parses each file to retrieve patient
    demographics and time series data (such as blood glucose, insulin infusion, and nutrition infusions), and
    compiles this information into a structured dictionary. It also collects summary statistics about the
    completeness of demographic fields and the availability of time series data, as well as any errors encountered
    during processing.

    Args:
        uploaded_file: A file-like object representing the uploaded zip file containing patient JSON files.

    Returns:
        patient_data (dict): A dictionary where each key is a patient ID and each value contains:
            - 'demographics': Patient demographic information (e.g., age, gender, diabetic status, weight, hospitalID)
            - 'time_series': Time series data for blood glucose, insulin infusion, enteral and parenteral nutrition
            - 'raw_filename': The original JSON filename
            - 'episode_count': Number of episodes in the JSON file
        patient_count (int): The number of patients successfully loaded.
        summary_stats (dict): Summary statistics including:
            - 'total_patients': Number of patients loaded
            - 'processing_errors': List of errors encountered during processing
            - 'demographic_completeness': Completeness statistics for each demographic field
            - 'time_series_availability': Availability statistics for each time series field

    Raises:
        Exception: If there is an error processing the zip file or its contents.

    Returns empty results if no file is provided or if an error occurs.
    """
    if uploaded_file is not None:
        try:
            import json

            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                json_files = []
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(".json"):
                            json_files.append(os.path.join(root, file))

                if not json_files:
                    raise Exception("No JSON files found in the uploaded ZIP")

                patient_data = {}
                processing_errors = []

                for json_file in json_files:
                    try:
                        patient_id = os.path.splitext(os.path.basename(json_file))[0]

                        with open(json_file, "r", encoding="utf-8") as f:
                            raw_data = json.load(f)

                        if "__class" not in raw_data or "episodes" not in raw_data:
                            processing_errors.append(
                                f"Invalid JSON structure in {os.path.basename(json_file)}"
                            )
                            continue

                        episodes = raw_data.get("episodes", [])
                        if not episodes:
                            processing_errors.append(
                                f"No episodes found in {os.path.basename(json_file)}"
                            )
                            continue

                        episode = episodes[0]

                        demographics = {}
                        if "diabeticStatus" in episode:
                            demographics["diabeticStatus"] = episode["diabeticStatus"]
                        if "gender" in episode:
                            demographics["gender"] = (
                                "Female" if episode["gender"] else "Male"
                            )
                        if "age" in episode:
                            demographics["age"] = episode["age"]
                        if "weight" in episode:
                            # Handle -1 as missing weight
                            weight = episode["weight"]
                            demographics["weight"] = None if weight == -1 else weight

                        if "hospitalID" in raw_data:
                            demographics["hospitalID"] = raw_data["hospitalID"]

                        time_series = {}

                        # Blood glucose measurements
                        if "bloodGlucose" in episode:
                            time_series["blood_glucose"] = episode["bloodGlucose"]

                        # Insulin infusion rate
                        if "insulinInfusion" in episode:
                            insulin_data = []
                            for entry in episode["insulinInfusion"]:
                                if isinstance(entry, list) and len(entry) >= 2:
                                    timestamp = entry[0]
                                    infusion_obj = entry[1]
                                    if (
                                        isinstance(infusion_obj, dict)
                                        and "rate" in infusion_obj
                                    ):
                                        insulin_data.append(
                                            [timestamp, infusion_obj["rate"]]
                                        )
                            time_series["insulin_infusion"] = insulin_data

                        # Nutrition infusion - separate enteral (type=0) and parenteral (type=1)
                        if "nutritionInfusion" in episode:
                            enteral_data = []
                            parenteral_data = []

                            for entry in episode["nutritionInfusion"]:
                                if isinstance(entry, list) and len(entry) >= 2:
                                    timestamp = entry[0]
                                    nutrition_obj = entry[1]
                                    if (
                                        isinstance(nutrition_obj, dict)
                                        and "rate" in nutrition_obj
                                        and "type" in nutrition_obj
                                    ):
                                        nutrition_type = nutrition_obj["type"]
                                        rate = nutrition_obj["rate"]

                                        if nutrition_type == 0:  # Enteral nutrition
                                            enteral_data.append([timestamp, rate])
                                        elif (
                                            nutrition_type == 1
                                        ):  # Parenteral nutrition
                                            parenteral_data.append([timestamp, rate])

                            time_series["enteral_nutrition"] = enteral_data
                            time_series["parenteral_nutrition"] = parenteral_data

                        patient_data[patient_id] = {
                            "demographics": demographics,
                            "time_series": time_series,
                            "raw_filename": os.path.basename(json_file),
                            "episode_count": len(episodes),
                        }

                    except Exception as e:
                        processing_errors.append(
                            f"Error processing {os.path.basename(json_file)}: {str(e)}"
                        )

                patient_count = len(patient_data)

                summary_stats = {
                    "total_patients": patient_count,
                    "processing_errors": processing_errors,
                    "demographic_completeness": {},
                    "time_series_availability": {},
                }

                if patient_count > 0:
                    demo_fields = [
                        "diabeticStatus",
                        "gender",
                        "age",
                        "weight",
                        "hospitalID",
                    ]
                    for field in demo_fields:
                        count = sum(
                            1
                            for p in patient_data.values()
                            if field in p["demographics"]
                            and p["demographics"][field] is not None
                        )
                        summary_stats["demographic_completeness"][field] = {
                            "count": count,
                            "percentage": (count / patient_count) * 100,
                        }

                    ts_fields = [
                        "blood_glucose",
                        "insulin_infusion",
                        "enteral_nutrition",
                        "parenteral_nutrition",
                    ]
                    for field in ts_fields:
                        count = sum(
                            1
                            for p in patient_data.values()
                            if field in p["time_series"]
                            and len(p["time_series"][field]) > 0
                        )
                        total_measurements = sum(
                            len(p["time_series"].get(field, []))
                            for p in patient_data.values()
                        )
                        summary_stats["time_series_availability"][field] = {
                            "patients_with_data": count,
                            "percentage": (count / patient_count) * 100,
                            "total_measurements": total_measurements,
                        }

                return patient_data, patient_count, summary_stats

        except Exception as e:
            raise Exception(f"Error processing UC3 zip file: {str(e)}")

    return {}, 0, {}


def save_example_data():
    data = """age,Sex,Pneumonia,PH,DiaPr,Respiratory rate,SPO2,GCS,SysPr,Pulse rate,SM PY,smoker,ex sm years,hospitalizations,(MT)
70,1,0,7.22,96,35,31,15,136,100,50,2,,,0
73,1,0,7.4,90,90,92,15,140,20,100,1,,,0
32,0,0,7.45,80,30,93,15,120,115,5,2,,,2
69,1,1,7.47,76,30,85,15,134,85,60,2,,,0
70,1,1,7.4,90,20,85,,180,74,60,2,,,0
52,0,0,7.42,86,28,92,,163,99,50,2,,,0
57,1,2,7.42,72,12,97,15,134,73,,3,,,2
84,0,0,7.33,,,,,,,,0,,,0
27,0,0,7.47,65,18,94,15,103,18,,4,,,0
70,0,0,7.47,,,,15,,120,,3,,,0"""

    with open("assets/example_data.csv", "w") as f:
        f.write(data)
