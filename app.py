import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config.use_case_config import USE_CASES
from src.data_loader import load_data, load_metadata, save_example_data
from src.quality_checks import run_all_checks
from src.rating import get_ratings, get_overall_rating
from src.uc1_image_quality_checks import run_all_checks_images, extract_patient_id_from_path

st.set_page_config(
    page_title="Standard Data Quality Framework",
    page_icon="📊",
    layout="wide"
)

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
    st.session_state.metadata = None
    st.session_state.selected_use_case = None
    st.session_state.ratings = None
    st.session_state.overall_rating = None
    st.session_state.target_column = None
    st.session_state.age_column = None
    st.session_state.height_column = None
    st.session_state.temp_data = None
    st.session_state.temp_metadata = None
    st.session_state.qualitative_scores = {}
    st.session_state.qualitative_ratings = None
    st.session_state.image_paths = None
    st.session_state.nrrd_directory = None

QUALITATIVE_DIMENSIONS = {
    "accessibility": {
        "name": "Accessibility",
        "definition": "Accessibility refers to the degree to which data is easily obtainable, clearly presented in a way that can be understood and available in a suitable format.",
        "question": "To what degree is the data easily obtainable, clearly presented, understandable, and available in a suitable format?",
        "options": [
            "1. Data cannot be located or retrieved.",
            "2. Data found only after substantial effort.",
            "3. Data located with moderate effort; format occasionally unclear.",
            "4. Data easily located; format mostly clear.",
            "5. Data immediately accessible; format crystal-clear and user-friendly."
        ]
    },
    "use_permissiveness": {
        "name": "Use Permissiveness",
        "definition": "Use permissiveness refers to the degree of permissiveness that licences and requirements for ethical and data governance approval grant to a user for the intended use.",
        "question": "To what degree are licences and governance requirements permissive for ethical and approved use of the data?",
        "options": [
            "1. Use prohibited or severely restricted.",
            "2. Multiple heavy constraints; licenses vague.",
            "3. Some permissions granted but with non-trivial hoops.",
            "4. Minor restrictions; licenses clear and largely permissive.",
            "5. Fully open use; licenses explicitly encourage broad reuse."
        ]
    },
    "availability": {
        "name": "Availability",
        "definition": "Availability refers to the degree to which data is present, obtainable and ready for use.",
        "question": "To what degree is the data present, obtainable, and ready for use?",
        "options": [
            "1. Data unavailable or offline.",
            "2. Downloadable link is provided but download fails or broken links.",
            "3. Occasional access issues without clear instruction or with formats issues.",
            "4. Mostly stable and clear downloadable links with common formats.",
            "5. Hosted on persistent, reliable repository with open, standard formats."
        ]
    },
    "compliance": {
        "name": "Compliance",
        "definition": "Compliance refers to the degree to which data has attributes that adhere to standards, conventions or regulations in force and similar rules relating to data quality in a specific context of use.",
        "question": "To what degree does the data adhere to relevant standards, conventions, and regulations in force and similar rules relating to data quality in a specific context of use?",
        "options": [
            "1. No adherence to any standards or regulations.",
            "2. Rough or partial compliance with many gaps.",
            "3. Met with core standards with some edge cases non-compliant.",
            "4. Satisfy all main regulations with minor deviations.",
            "5. Fully compliant with all relevant standards/laws."
        ]
    },
    "provenance": {
        "name": "Provenance",
        "definition": "Provenance means a description of the source of the original or raw data prior to any subsequent processing or transformation, including context, purpose, method and technology of data generation.",
        "question": "To what degree is the source, context, purpose, method, and technology of data generation documented?",
        "options": [
            "1. No source or collection details.",
            "2. Only source name is provided with no methods or context.",
            "3. Basic method description is provided but dates, operators, organizations, and other key components are missing.",
            "4. Methods, purpose, dates, and other key components are provided but missing minor context.",
            "5. Complete provenance is provided with detailed descriptions of method, system, dates, agents, data processing process."
        ]
    },
    "trustworthiness": {
        "name": "Trustworthiness",
        "definition": "Trustworthiness refers to the degree to which context information fosters trust in the data, e.g. by documenting agents involved in the provenance of data, data validation routines, the provenance of available provenance information etc.",
        "question": "To what degree does documentation of agents, validation routines, and provenance foster trust in the dataset?",
        "options": [
            "1. No information about data origin, agents, or validation.",
            "2. Only some provenance or agent names mentioned, but no details on processes.",
            "3. Key agents and data-validation steps are named, but provenance chains are incomplete.",
            "4. Clear documentation of most agents, validation routines, and where provenance logs exist.",
            "5. Comprehensive records of all agents, end-to-end validation processes, and detailed provenance metadata."
        ]
    },
    "consistency": {
        "name": "Consistency",
        "definition": "Consistency refers to the degree to which data has attributes that are free from contradiction and are coherent with other data in a specific context of use. It can be either or both among data regarding one entity and across similar data for comparable entities.",
        "question": "To what degree is the data free from internal contradictions and logically coherent?",
        "options": [
            "1. Logical contradictions pervasive.",
            "2. Frequent logical errors in data attributes (age>200 yrs old, male with pregnancy label).",
            "3. Occasional inconsistencies in data attributes.",
            "4. Rare inconsistencies.",
            "5. Fully consistent throughout."
        ]
    }
}


def create_assets_dir():
    if not os.path.exists("assets"):
        os.makedirs("assets")


def save_qualitative_template():
    template = {
        "qualitative_assessment": {
            "accessibility": 0,
            "use_permissiveness": 0,
            "availability": 0,
            "compliance": 0,
            "provenance": 0,
            "trustworthiness": 0,
            "consistency": 0
        },
        "instructions": "Replace 0 with your score (1-5) for each dimension"
    }

    with open("assets/qualitative_template.json", "w") as f:
        json.dump(template, f, indent=2)


def display_metrics(ratings, metric_type="Quantitative"):
    metric_names = {
        "population_representativity": "Population Representativity",
        "metadata_granularity": "Metadata Granularity",
        "accuracy": "Accuracy",
        "coherence": "Coherence",
        "semantic_coherence": "Semantic Coherence",
        "completeness": "Completeness",
        "relational_consistency": "Relational Consistency",
        "clinical_stage_consistency": "Clinical Stage Consistency",
        "accessibility": "Accessibility",
        "use_permissiveness": "Use Permissiveness",
        "availability": "Availability",
        "compliance": "Compliance",
        "provenance": "Provenance",
        "trustworthiness": "Trustworthiness",
        "consistency": "Consistency"
    }

    calculation_methods = {
        "population_representativity": "Score based on how close class distribution is to ideal balance",
        "metadata_granularity": "patients with metadata / total patients",
        "accuracy": "values within expected range / total values checked",
        "coherence": "features with consistent data types / total features",
        "semantic_coherence": "unique column names / total columns",
        "completeness": "non-missing values / total values",
        "relational_consistency": "unique rows / total rows",
        "accessibility": "User assessment based on data obtainability and clarity",
        "use_permissiveness": "User assessment based on license permissiveness",
        "availability": "User assessment based on data availability and access",
        "compliance": "User assessment based on standards adherence",
        "provenance": "User assessment based on documentation completeness",
        "trustworthiness": "User assessment based on trust indicators",
        "consistency": "User assessment based on logical coherence"
    }

    calculation_methods_images = {
        "population_representativity": "Minority/majority class ratio from histology distribution (≤0.2 ratio = 1/5, ≥0.8 ratio = 5/5)",
        "metadata_granularity": "Patients with complete metadata / total patients (≤0.2 ratio = 1/5, ≥0.8 ratio = 5/5)",
        "accuracy": "Combined score: slice dimension consistency + missing slice detection (≤0.2 error ratio = 5/5, ≥0.8 error ratio = 1/5)",
        "coherence": "Number of channels consistency across images (≤0.2 inconsistency = 5/5, ≥0.8 inconsistency = 1/5)",
        "semantic_coherence": "Duplicate image detection using array hash comparison (≤0.2 duplication = 5/5, ≥0.8 duplication = 1/5)",
        "completeness": "Missing pixels / total pixels ratio across all images (≤0.2 missing = 5/5, ≥0.8 missing = 1/5)",
        "relational_consistency": "File duplication + patient ID format consistency (≤0.2 issues = 5/5, ≥0.8 issues = 1/5)"
    }

    rating_thresholds = """
        - Value 0.0-0.2: Rating 1/5
        - Value 0.2-0.4: Rating 2/5
        - Value 0.4-0.6: Rating 3/5
        - Value 0.6-0.8: Rating 4/5
        - Value 0.8-1.0: Rating 5/5
        """

    is_image_data = st.session_state.selected_use_case == "Use case 1" and st.session_state.image_paths is not None

    for metric, (rating, value, explanation) in ratings.items():
        display_name = metric_names.get(metric, metric)

        with st.expander(f"{display_name}: {rating}/5", expanded=True):
            cols = st.columns([2, 1])

            with cols[0]:
                st.markdown("**Explanation:**")
                st.write(explanation)

                st.markdown("**Calculation Method:**")
                if is_image_data and metric in calculation_methods_images:
                    st.write(calculation_methods_images.get(metric))
                else:
                    st.write(calculation_methods.get(metric, "No calculation method available"))

                st.markdown("**Rating Thresholds:**")
                st.markdown(rating_thresholds)

            with cols[1]:
                fig = px.pie(values=[rating, 5 - rating], names=["Score", "Remaining"],
                             hole=0.7, color_discrete_sequence=["#1f77b4", "#e0e0e0"])
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    annotations=[dict(text=f"{rating}/5", x=0.5, y=0.5, font_size=20, showarrow=False)]
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{metric_type}_{metric}_pie")

                st.markdown(f"**Raw Value:** {value:.3f}")
                st.markdown("_Higher values are better for all metrics_")


def display_radar_chart(ratings, title="Quality Ratings Radar Chart"):
    metrics = []
    values = []

    for metric, (rating, _, _) in ratings.items():
        metric_names = {
            "population_representativity": "Pop. Representativity",
            "metadata_granularity": "Metadata Granularity",
            "accuracy": "Accuracy",
            "coherence": "Coherence",
            "semantic_coherence": "Semantic Coherence",
            "completeness": "Completeness",
            "relational_consistency": "Relational Consistency",
            "clinical_stage_consistency": "Clinical Staging",
            "accessibility": "Accessibility",
            "use_permissiveness": "Use Permissiveness",
            "availability": "Availability",
            "compliance": "Compliance",
            "provenance": "Provenance",
            "trustworthiness": "Trustworthiness",
            "consistency": "Consistency"
        }

        display_name = metric_names.get(metric, metric)
        metrics.append(display_name)
        values.append(rating)

    fig = px.line_polar(
        r=values,
        theta=metrics,
        line_close=True,
        range_r=[0, 5],
        title=title
    )

    fig.update_traces(fill='toself')
    st.plotly_chart(fig, use_container_width=True, key=f"radar_{title.replace(' ', '_').lower()}")


def calculate_qualitative_ratings(scores):
    ratings = {}
    for dimension, score in scores.items():
        if dimension in QUALITATIVE_DIMENSIONS:
            value = (score - 1) / 4
            explanation = f"User selected option {score}: {QUALITATIVE_DIMENSIONS[dimension]['options'][score - 1]}"
            ratings[dimension] = (score, value, explanation)
    return ratings


def get_use_case_specific_columns(selected_use_case, data_columns):
    """Get appropriate column suggestions based on use case and available data."""
    use_case_info = USE_CASES.get(selected_use_case, {})

    if selected_use_case == "Use case 1":
        target_options = ["deadstatus.event", "Overall.Stage"]
        age_options = ["age"]
        other_options = ["Survival.time"]
    elif selected_use_case == "Use case 4":
        target_options = ["(MT)"]
        age_options = ["age"]
        other_options = ["Respiratory rate", "SPO2", "SysPr", "Pulse rate"]
    else:
        target_options = []
        age_options = []
        other_options = []

    available_target = [col for col in target_options if col in data_columns]
    available_age = [col for col in age_options if col in data_columns]
    available_other = [col for col in other_options if col in data_columns]

    return available_target, available_age, available_other


def load_nrrd_directory(directory_path):
    """
    Load all NRRD files from a directory.

    :param directory_path: Path to directory containing NRRD files
    :return: List of NRRD file paths
    :rtype: list
    """
    nrrd_files = []

    # Add more patterns below if files have different extensions
    patterns = [
        os.path.join(directory_path, '**', '*.nrrd'),
    ]

    for pattern in patterns:
        nrrd_files.extend(glob.glob(pattern, recursive=True))

    # Prioritize image files over mask files
    image_files = [f for f in nrrd_files if 'image' in os.path.basename(f).lower()]
    if not image_files:
        # If no specific image files, take all NRRD files
        image_files = nrrd_files

    return image_files


def main():
    create_assets_dir()

    if not os.path.exists("assets/example_data.csv"):
        save_example_data()

    if not os.path.exists("assets/qualitative_template.json"):
        save_qualitative_template()

    st.title("Standard Data Quality Framework")

    st.sidebar.title("Use Case Selection")
    selected_use_case = st.sidebar.selectbox(
        "Select a Use Case",
        list(USE_CASES.keys()),
        index=0,
        key="use_case_selector"
    )

    use_case_info = USE_CASES[selected_use_case]

    st.sidebar.markdown(f"### {use_case_info['name']}")
    st.sidebar.markdown(f"*{use_case_info['description']}*")

    if not use_case_info['implemented']:
        st.warning(
            f"⚠️ {selected_use_case} is not yet implemented in this POC. Only Use Case 1 (DuneAI) and Use Case 4 (ASCOPD) are currently functional.")

    st.sidebar.markdown("---")

    if selected_use_case == "Use case 1":
        st.sidebar.markdown("### UC1 Image Data Input")
        nrrd_directory = st.sidebar.text_input(
            "Path to NRRD files directory",
            value="assets/converted_nrrds",
            help="Enter the path to the directory containing NRRD image files"
        )

        uploaded_clinical_csv = st.sidebar.file_uploader(
            "Upload clinical CSV file (optional)",
            type="csv",
            help="Upload NSCLC-Radiomics clinical CSV file for enhanced analysis"
        )

        if st.sidebar.button("Load Data"):
            if os.path.exists(nrrd_directory):
                image_paths = load_nrrd_directory(nrrd_directory)
                if image_paths:
                    st.session_state.image_paths = image_paths
                    st.session_state.nrrd_directory = nrrd_directory
                    st.session_state.selected_use_case = selected_use_case

                    # Try to load clinical metadata from uploaded file
                    if uploaded_clinical_csv is not None:
                        try:
                            clinical_data = load_metadata(uploaded_clinical_csv)
                            if clinical_data is not None:
                                st.session_state.metadata = clinical_data
                                st.success(
                                    f"Loaded {len(image_paths)} image files and clinical data for {len(clinical_data)} patients!")
                            else:
                                st.warning(f"Loaded {len(image_paths)} image files but could not load clinical data.")
                        except Exception as e:
                            st.error(f"Error loading clinical metadata: {str(e)}")
                            st.success(f"Loaded {len(image_paths)} image files (without clinical data)")
                    else:
                        st.session_state.metadata = None
                        st.success(f"Loaded {len(image_paths)} image files (no clinical data provided)")
                else:
                    st.error("No NRRD/MHA files found in the specified directory!")
            else:
                st.error("Directory does not exist! Please check the path.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload your dataset", type="csv")
        uploaded_metadata = st.sidebar.file_uploader("Upload metadata (optional)", type="csv")

        if uploaded_file is not None:
            try:
                st.session_state.temp_data = load_data(uploaded_file)
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.session_state.temp_data = None

        if uploaded_metadata is not None:
            try:
                st.session_state.temp_metadata = load_metadata(uploaded_metadata)
            except Exception as e:
                st.error(f"Error loading metadata: {str(e)}")
                st.session_state.temp_metadata = None

        if st.sidebar.button("Load Data"):
            if st.session_state.temp_data is not None:
                st.session_state.processed_data = st.session_state.temp_data
                st.session_state.metadata = st.session_state.temp_metadata
                st.session_state.selected_use_case = selected_use_case
                st.session_state.image_paths = None
                st.success("Data loaded successfully!")
            else:
                st.error("Please upload a dataset first!")

    if (st.session_state.processed_data is not None) or (st.session_state.image_paths is not None):
        if st.session_state.image_paths is not None:
            st.subheader("Image Data Overview")
            st.write(f"**Total images loaded:** {len(st.session_state.image_paths)}")

            patient_ids = set()
            for path in st.session_state.image_paths:
                patient_id = extract_patient_id_from_path(path)
                patient_ids.add(patient_id)

            st.write(f"**Number of patients:** {len(patient_ids)}")
            st.write(f"**Average images per patient:** {len(st.session_state.image_paths) / len(patient_ids):.1f}")

            if st.session_state.metadata is not None:
                st.write(f"**Clinical metadata available for:** {len(st.session_state.metadata)} patients")

                # Show clinical data coverage
                clinical_patient_ids = set(st.session_state.metadata['PatientID'].tolist())
                coverage = len(patient_ids.intersection(clinical_patient_ids)) / len(patient_ids) * 100
                st.write(f"**Clinical data coverage:** {coverage:.1f}% of image patients")

            with st.expander("Sample image paths"):
                for i, path in enumerate(st.session_state.image_paths[:10]):
                    st.text(f"{i + 1}. {Path(path).name}")
                if len(st.session_state.image_paths) > 10:
                    st.text(f"... and {len(st.session_state.image_paths) - 10} more files")

            # Clinical Data Overview Section
            if st.session_state.metadata is not None:
                st.subheader("Clinical Data Overview")

                # Basic statistics
                total_patients = len(st.session_state.metadata)
                total_columns = len(st.session_state.metadata.columns)
                st.write(f"**Total patients in clinical data:** {total_patients}")
                st.write(f"**Total clinical fields:** {total_columns}")

                # Data completeness analysis
                non_missing_counts = st.session_state.metadata.count()
                overall_completeness = (non_missing_counts.sum() / (total_patients * total_columns)) * 100
                st.write(f"**Overall data completeness:** {overall_completeness:.1f}%")

                # Key field analysis
                key_fields = ['age', 'gender', 'Histology', 'Overall.Stage', 'Survival.time', 'deadstatus.event']
                available_key_fields = [field for field in key_fields if field in st.session_state.metadata.columns]

                if available_key_fields:
                    st.write(f"**Key fields available:** {len(available_key_fields)}/{len(key_fields)}")

                    # Show completeness for key fields
                    key_field_completeness = []
                    for field in available_key_fields:
                        completeness = (st.session_state.metadata[field].count() / total_patients) * 100
                        key_field_completeness.append(f"{field}: {completeness:.1f}%")

                    st.write(f"**Key field completeness:** {', '.join(key_field_completeness[:3])}")
                    if len(key_field_completeness) > 3:
                        st.write(f"**Additional fields:** {', '.join(key_field_completeness[3:])}")

                # Histology distribution
                if 'Histology' in st.session_state.metadata.columns:
                    histology_counts = st.session_state.metadata['Histology'].value_counts()
                    most_common = histology_counts.head(3)
                    histology_summary = []
                    for hist_type, count in most_common.items():
                        if pd.notna(hist_type):
                            percentage = (count / total_patients) * 100
                            histology_summary.append(f"{hist_type}: {count} ({percentage:.1f}%)")

                    if histology_summary:
                        st.write(f"**Top histology types:** {', '.join(histology_summary)}")

                # Age statistics
                if 'age' in st.session_state.metadata.columns:
                    age_data = pd.to_numeric(st.session_state.metadata['age'], errors='coerce').dropna()
                    if len(age_data) > 0:
                        mean_age = age_data.mean()
                        age_range = f"{age_data.min():.1f}-{age_data.max():.1f}"
                        st.write(f"**Age statistics:** Mean {mean_age:.1f} years, Range {age_range} years")

                with st.expander("Clinical data columns"):
                    cols_per_row = 3
                    columns = list(st.session_state.metadata.columns)
                    for i in range(0, len(columns), cols_per_row):
                        row_cols = st.columns(cols_per_row)
                        for j, col_name in enumerate(columns[i:i + cols_per_row]):
                            with row_cols[j]:
                                completeness = (st.session_state.metadata[col_name].count() / total_patients) * 100
                                st.text(f"{col_name} ({completeness:.0f}%)")

        else:
            st.subheader("Data Preview")
            st.dataframe(st.session_state.processed_data, use_container_width=True)

        st.markdown("---")
        st.subheader("Qualitative Check Configuration")

        col1, col2 = st.columns([1, 3])
        with col1:
            qualitative_file = st.file_uploader("Upload qualitative assessment (JSON)", type="json")

            with open("assets/qualitative_template.json", "rb") as f:
                st.download_button(
                    label="Download JSON Template",
                    data=f,
                    file_name="qualitative_template.json",
                    mime="application/json"
                )

        with col2:
            if qualitative_file is not None:
                try:
                    qualitative_data = json.load(qualitative_file)
                    if "qualitative_assessment" in qualitative_data:
                        st.session_state.qualitative_scores = qualitative_data["qualitative_assessment"]
                        st.success("Qualitative assessment loaded from JSON!")
                except Exception as e:
                    st.error(f"Error loading qualitative assessment: {str(e)}")

        st.markdown("### Manual Qualitative Assessment")
        st.markdown("Please answer the following questions about your dataset:")

        for key, dimension in QUALITATIVE_DIMENSIONS.items():
            st.markdown(f"**{dimension['name']}**")
            st.markdown(f"*{dimension['definition']}*")

            if key not in st.session_state.qualitative_scores:
                st.session_state.qualitative_scores[key] = 1

            current_score = st.session_state.qualitative_scores.get(key, 1)
            selected_index = current_score - 1 if current_score > 0 else 0

            selected_option = st.radio(
                dimension['question'],
                options=dimension['options'],
                key=f"qual_{key}",
                index=selected_index
            )

            score = int(selected_option[0])
            st.session_state.qualitative_scores[key] = score

        if st.session_state.image_paths is None:
            st.markdown("---")
            st.subheader("Quantitative Check Configuration")

            data_columns = list(st.session_state.processed_data.columns)
            available_target, available_age, available_other = get_use_case_specific_columns(
                st.session_state.selected_use_case, data_columns)

            target_column = st.selectbox(
                "Select Target Column",
                options=["None"] + (available_target if available_target else data_columns),
                help="Select the column to be used as the target variable for calculating population representativity.",
                key="target_column_selector"
            )

            age_column = st.selectbox(
                "Select Age Column",
                options=["None"] + (available_age if available_age else data_columns),
                help="Select the column containing age data for accuracy checks (expected range: 0-120 years).",
                key="age_column_selector"
            )

            other_numeric_column = st.selectbox(
                "Select Additional Numeric Column for Validation",
                options=["None"] + (available_other if available_other else data_columns),
                help="Select another numeric column for accuracy validation (ranges depend on use case).",
                key="other_column_selector"
            )

            st.session_state.target_column = target_column if target_column != "None" else None
            st.session_state.age_column = age_column if age_column != "None" else None
            st.session_state.other_column = other_numeric_column if other_numeric_column != "None" else None

        if st.button("Run Quality Checks", type="primary"):
            if all(score > 0 for score in st.session_state.qualitative_scores.values()):
                if not USE_CASES[st.session_state.selected_use_case]['implemented']:
                    st.error(
                        f"⚠️ {st.session_state.selected_use_case} is not yet implemented. Please select Use Case 1 or Use Case 4.")
                else:
                    with st.spinner("Running quality checks..."):
                        if st.session_state.image_paths is not None:
                            check_results = run_all_checks_images(
                                st.session_state.image_paths,
                                st.session_state.metadata
                            )
                        else:
                            use_case_config = USE_CASES[st.session_state.selected_use_case].copy()
                            use_case_config["target_column"] = st.session_state.target_column

                            expected_ranges = use_case_config.get("expected_ranges", {}).copy()
                            if st.session_state.age_column and st.session_state.age_column not in expected_ranges:
                                expected_ranges[st.session_state.age_column] = [0, 120]

                            use_case_config["expected_ranges"] = expected_ranges
                            use_case_config["age_column"] = st.session_state.age_column
                            use_case_config["other_column"] = st.session_state.other_column

                            check_results = run_all_checks(
                                st.session_state.processed_data,
                                use_case_config,
                                st.session_state.metadata
                            )

                        st.session_state.ratings = get_ratings(check_results)
                        st.session_state.overall_rating = get_overall_rating(st.session_state.ratings)

                        st.session_state.qualitative_ratings = calculate_qualitative_ratings(
                            st.session_state.qualitative_scores)

                    st.success("Quality checks completed!")
            else:
                st.error("Please complete all qualitative assessment questions before running quality checks!")

        if st.session_state.ratings is not None and st.session_state.qualitative_ratings is not None:
            st.markdown("---")
            st.subheader("Quality Assessment Results")
            if st.session_state.selected_use_case == "Use case 1":
                st.info("📊 **Note**: UC1 quantitative scores follow SDQF guidelines. Specific thresholds (≤0.2, ≥0.8) determine 1-5 scale ratings.")

            col1, col2 = st.columns(2)

            with col1:
                display_radar_chart(st.session_state.qualitative_ratings, "Qualitative Assessment")
                qualitative_avg = np.mean([r[0] for r in st.session_state.qualitative_ratings.values()])
                st.markdown(f"**Average Qualitative Score: {qualitative_avg:.1f}/5**")

            with col2:
                display_radar_chart(st.session_state.ratings, "Quantitative Assessment")
                st.markdown(f"**Average Quantitative Score: {st.session_state.overall_rating:.1f}/5**")

            total_avg = (qualitative_avg + st.session_state.overall_rating) / 2
            st.markdown("---")
            st.markdown(f"### Overall Data Quality Score: {total_avg:.1f}/5")

            st.markdown("---")
            st.subheader("Detailed Qualitative Scores")
            display_metrics(st.session_state.qualitative_ratings, "Qualitative")

            st.markdown("---")
            st.subheader("Detailed Quantitative Scores")
            display_metrics(st.session_state.ratings, "Quantitative")

    else:
        if selected_use_case == "Use case 1":
            st.info("Please enter the path to your NRRD files directory and click 'Load Image Data' to begin.")
        else:
            st.info("Please upload a dataset and click 'Load Data' to begin.")


if __name__ == "__main__":
    main()
