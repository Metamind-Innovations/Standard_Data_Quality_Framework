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
                "description": "Distribution balance of histology types from clinical data or patient image distribution",
                "method": "histology_distribution_analysis"
            },
            "metadata_granularity": {
                "description": "Combined assessment of embedded DICOM metadata and clinical data coverage",
                "method": "metadata_coverage_analysis"
            },
            "accuracy": {
                "description": "Image quality validation (HU ranges, spacing, dimensions) and clinical data accuracy",
                "method": "combined_validation_checks"
            },
            "coherence": {
                "description": "Consistency of image properties across the dataset",
                "method": "image_property_consistency"
            },
            "semantic_coherence": {
                "description": "Patient ID consistency and naming pattern validation",
                "method": "semantic_validation"
            },
            "completeness": {
                "description": "Image volume completeness and clinical data availability",
                "method": "data_completeness_analysis"
            },
            "relational_consistency": {
                "description": "Duplicate detection and patient ID format consistency",
                "method": "relationship_validation"
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
