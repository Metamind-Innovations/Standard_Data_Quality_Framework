USE_CASES = {
    "Use case 1": {
        "name": "DuneAI",
        "description": "Lung Cancer medical algorithm evaluation supporting EHR and imaging data",
        "implemented": False
    },
    "Use case 2": {
        "name": "PGx2P",
        "description": "Pharmacogenomics Passports to Practice",
        "implemented": False
    },
    "Use case 3": {
        "name": "STAR",
        "description": "Glucose control in intensive care units",
        "implemented": False
    },
    "Use case 4": {
        "name": "ASCOPD",
        "description": "Medical AI models for COPD and ASTHMA inpatient risk stratification",
        "implemented": True,
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
        "implemented": False
    }
}