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
    Load and process UC2 zip file containing VCF files and CSV ground truth.

    Returns:
        groundtruth_df: DataFrame containing ground truth phenotype data
        vcf_dataframes: List of DataFrames containing VCF data (after ## lines)
        vcf_metadata: List of metadata strings for each VCF file (## lines)
        vcf_filenames: List of VCF filenames (basename only)
    """
    if uploaded_file is not None:
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract zip file
                with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find VCF and CSV files
                vcf_files = []
                csv_files = []

                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith(".vcf") or file.endswith(".vcf.gz"):
                            vcf_files.append(file_path)
                        elif file.endswith(".csv"):
                            csv_files.append(file_path)

                # Load ground truth CSV
                groundtruth_df = None
                if csv_files:
                    # Look for groundtruth phenotype file or take the first CSV
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
                    # Store the filename (basename only)
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
                            # This is the header line for the data
                            data_start_idx = i
                            break

                    # Store metadata as list of strings
                    vcf_metadata.append(metadata_lines)

                    # Extract data portion (from #CHROM line onwards)
                    if data_start_idx is not None:
                        # Get header
                        header_line = lines[data_start_idx].strip().lstrip("#")
                        headers = header_line.split("\t")

                        # Get data lines
                        data_lines = []
                        for line in lines[data_start_idx + 1 :]:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                data_lines.append(line.split("\t"))

                        # Create DataFrame
                        if data_lines:
                            vcf_df = pd.DataFrame(data_lines, columns=headers)
                            vcf_dataframes.append(vcf_df)
                        else:
                            # Empty VCF file - create empty DataFrame with headers
                            vcf_df = pd.DataFrame(columns=headers)
                            vcf_dataframes.append(vcf_df)
                    else:
                        # No data section found - create empty DataFrame
                        vcf_df = pd.DataFrame()
                        vcf_dataframes.append(vcf_df)

                return groundtruth_df, vcf_dataframes, vcf_metadata, vcf_filenames

        except Exception as e:
            raise Exception(f"Error processing UC2 zip file: {str(e)}")

    return None, [], [], []


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
