# Standard Data Quality Framework (SDQF)

A comprehensive Streamlit-based web application for assessing data quality across multiple dimensions using both
qualitative and quantitative metrics, with specialized support for medical imaging datasets.

## 🚀 Features

### Dual Assessment Approach

- **Qualitative Expert Judgment**: Interactive questionnaire for subjective quality assessment
- **Quantitative Automated Checks**: Algorithm-based validation with configurable thresholds
- **Multi-Modal Support**: Handles both tabular data and medical imaging datasets (NRRD, MHA formats)

### Interactive Web Interface

- User-friendly Streamlit application with visual feedback
- Real-time quality assessment with radar charts and detailed metrics
- JSON-based configuration for qualitative assessments
- Extensible framework supporting multiple healthcare use cases

## 📊 Quality Dimensions Detail

### Quantitative Metrics (Automated)

1. **Population Representativity**: Class balance analysis in target variables
2. **Metadata Granularity**: Ratio of records with complete metadata
3. **Accuracy**: Proportion of values within expected clinical ranges
4. **Coherence**: Data type consistency across features
5. **Semantic Coherence**: Column name uniqueness and value consistency
6. **Completeness**: Proportion of non-missing values
7. **Relational Consistency**: Data row uniqueness validation

### Qualitative Metrics (Expert Assessment)

1. **Accessibility**: Ease of data retrieval and comprehension
2. **Use Permissiveness**: License and governance flexibility
3. **Availability**: Data presence and readiness for use
4. **Compliance**: Adherence to standards and regulations
5. **Provenance**: Documentation of data source and generation
6. **Trustworthiness**: Trust indicators in documentation
7. **Consistency**: Logical coherence and contradiction-free data

### UC1-Specific Image Quality Metrics

1. **Population Representativity**: Histology distribution balance or patient image count consistency
2. **Metadata Granularity**: Clinical metadata coverage across imaging patients
3. **Accuracy**: CT slice dimension consistency and completeness validation
4. **Coherence**: Image channel consistency (grayscale vs RGB)
5. **Semantic Coherence**: Duplicate image detection using content hashing
6. **Completeness**: Missing pixel analysis across all images
7. **Relational Consistency**: File integrity and patient ID format validation

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/standard-data-quality-framework.git
cd standard-data-quality-framework
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚦 Getting Started

### Basic Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Follow the workflow:
    - **Select a use case** (UC1: DuneAI or UC4: ASCOPD)
    - **Upload your dataset** (CSV for tabular data, NRRD directory path for images)
    - **Optionally upload metadata** CSV
    - **Complete qualitative assessment** questionnaire
    - **Configure quantitative checks** (select target, age, and validation columns)
    - **Run quality checks** and review results

### Use Case 1 (DuneAI) - Image Data Workflow

1. **Prepare NRRD files**: Ensure CT scans are in NRRD format in organized directories
2. **Clinical metadata**: Upload NSCLC-Radiomics clinical CSV file (optional)
3. **Directory path**: Enter path to NRRD files directory (e.g., `assets/converted_nrrds`)
4. **Load data**: Click "Load Data" to process images and extract patient information
5. **Assessment**: Complete qualitative questionnaire and run quantitative checks
6. **Results**: Review image-specific quality metrics and clinical correlations

### Use Case 4 (ASCOPD) - Tabular Data Workflow

1. **Upload CSV**: Select your COPD/Asthma patient dataset
2. **Metadata** (optional): Upload additional metadata CSV
3. **Column selection**: Choose target variable, age column, and validation columns
4. **Assessment**: Complete qualitative questionnaire
5. **Run checks**: Execute both qualitative and quantitative assessments
6. **Results**: Review comprehensive quality analysis with clinical range validation

## 📁 Project Structure

```
standard-data-quality-framework/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── assets/                         # Sample data and templates
│   ├── example_data.csv           # Sample ASCOPD dataset
│   └── qualitative_template.json  # Qualitative assessment template
├── config/                         # Configuration files
│   └── use_case_config.py         # Use case definitions and parameters
├── src/                           # Source code modules
│   ├── __init__.py                # UC1 image quality checks exports
│   ├── data_loader.py             # Data loading utilities
│   ├── quality_checks.py          # Tabular data quality checks
│   ├── rating.py                  # Rating calculation system
│   └── uc1_image_quality_checks.py # Image quality assessment functions
└── test_data_radiomics/           # Sample UC1 clinical data
    └── NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv
```

## 📈 Scoring System

All metrics use a consistent **1-5 rating scale**:

- **1/5**: Poor quality (0-20% score)
- **2/5**: Below average (20-40% score)
- **3/5**: Average (40-60% score)
- **4/5**: Good (60-80% score)
- **5/5**: Excellent (80-100% score)

### UC1 SDQF Guidelines

Image quality metrics follow specific SDQF thresholds:

- **≤0.2 error/inconsistency ratio**: 5/5 rating
- **≥0.8 error/inconsistency ratio**: 1/5 rating
- **Linear interpolation** for intermediate values

## 📜 License
All rights reserved by MetaMinds Innovations.
