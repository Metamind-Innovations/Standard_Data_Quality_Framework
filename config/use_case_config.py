USE_CASES = {
    "Use case 1": {
        "name": "DuneAI",
        "description": "Lung Cancer medical algorithm evaluation supporting EHR and imaging data",
        "implemented": True,
        "data_type": "images",
        "expected_columns": [
            "PatientID", "age", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage",
            "Overall.Stage", "Histology", "gender", "Survival.time", "deadstatus.event"
        ],
        "column_types": {
            "PatientID": "categorical",
            "age": "numeric",
            "clinical.T.Stage": "categorical",
            "Clinical.N.Stage": "categorical",
            "Clinical.M.Stage": "categorical",
            "Overall.Stage": "categorical",
            "Histology": "categorical",
            "gender": "categorical",
            "Survival.time": "numeric",
            "deadstatus.event": "categorical"
        },
        "expected_ranges": {
            "age": [0, 120],
            "clinical.T.Stage": [1, 5],
            "Clinical.N.Stage": [0, 4],
            "Clinical.M.Stage": [0, 3],
            "Survival.time": [0, 10000]
        },
        "target_column": "deadstatus.event",
        "image_quality_checks": {
            "population_representativity": {
                "description": "Minority/majority class ratio from histology distribution or patient image count balance",
                "method": "minority_majority_ratio_analysis",
                "sdqf_rating": "≤0.2 ratio = 1/5, ≥0.8 ratio = 5/5"
            },
            "metadata_granularity": {
                "description": "Patients with complete metadata / total patients ratio",
                "method": "metadata_coverage_ratio_analysis",
                "sdqf_rating": "≤0.2 ratio = 1/5, ≥0.8 ratio = 5/5"
            },
            "accuracy": {
                "description": "CT-scan slice dimension consistency and missing slice detection",
                "method": "slice_consistency_and_completeness_analysis",
                "sdqf_rating": "≤0.2 error ratio = 5/5, ≥0.8 error ratio = 1/5"
            },
            "coherence": {
                "description": "Consistent number of channels across all images (grayscale vs RGB)",
                "method": "channel_consistency_analysis",
                "sdqf_rating": "≤0.2 inconsistency ratio = 5/5, ≥0.8 inconsistency ratio = 1/5"
            },
            "semantic_coherence": {
                "description": "Duplicate image detection using array hash comparison",
                "method": "image_duplicate_detection",
                "sdqf_rating": "≤0.2 duplication ratio = 5/5, ≥0.8 duplication ratio = 1/5"
            },
            "completeness": {
                "description": "Missing pixels / total pixels ratio across all images",
                "method": "pixel_completeness_analysis",
                "sdqf_rating": "≤0.2 missing ratio = 5/5, ≥0.8 missing ratio = 1/5"
            },
            "relational_consistency": {
                "description": "File duplication and patient ID format consistency checks",
                "method": "file_and_id_consistency_validation",
                "sdqf_rating": "≤0.2 issues ratio = 5/5, ≥0.8 issues ratio = 1/5"
            }
        },
        "clinical_metadata_file": "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv",
        "image_extensions": [".nrrd", ".mha", ".nii", ".nii.gz"],
        "expected_patient_id_pattern": "LUNG1-\\d{3}"
    },
    "Use case 2": {
        "name": "PGx2P",
        "description": "Pharmacogenomics Passports to Practice",
        "implemented": False,
        "data_type": "tabular"
    },
    "Use case 3": {
        "name": "STAR",
        "description": "Glucose control in intensive care units",
        "implemented": False,
        "data_type": "time_series"
    },
    "Use case 4": {
        "name": "ASCOPD",
        "description": "Medical AI models for COPD and ASTHMA inpatient risk stratification",
        "implemented": True,
        "data_type": "tabular",
        "expected_columns": [
            "age", "Sex", "Pneumonia", "PH", "DiaPr", "Respiratory rate",
            "SPO2", "GCS", "SysPr", "Pulse rate", "SM PY", "smoker",
            "ex sm years", "hospitalizations", "(MT)"
        ],
        "column_types": {
            "age": "numeric",
            "Sex": "categorical",
            "Pneumonia": "categorical",
            "PH": "numeric",
            "DiaPr": "numeric",
            "Respiratory rate": "numeric",
            "SPO2": "numeric",
            "GCS": "numeric",
            "SysPr": "numeric",
            "Pulse rate": "numeric",
            "SM PY": "numeric",
            "smoker": "categorical",
            "ex sm years": "numeric",
            "hospitalizations": "numeric",
            "(MT)": "categorical"
        },
        "expected_ranges": {
            "age": [0, 120],
            "PH": [7.0, 7.7],
            "DiaPr": [40, 120],
            "Respiratory rate": [8, 60],
            "SPO2": [70, 100],
            "GCS": [3, 15],
            "SysPr": [70, 220],
            "Pulse rate": [40, 180],
            "SM PY": [0, 150]
        },
        "target_column": "(MT)"
    },
    "Use case 5": {
        "name": "COPowereD",
        "description": "Medical AI algorithms using patient reported outcomes for COPD to detect and predict hospitalization or acute exacerbations",
        "implemented": False,
        "data_type": "tabular"
    }
}
