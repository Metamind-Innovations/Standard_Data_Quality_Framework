Standard Data Quality Framework
A framework for assessing and ensuring data quality across medical AI applications.

Overview
This tool provides a standardized approach to evaluating data quality across different medical AI use cases. It performs automated checks on uploaded datasets and provides quality ratings based on various dimensions.

Use Cases
The framework supports the following use cases:

DuneAI: Lung Cancer medical algorithm evaluation supporting EHR and imaging data
PGx2P: Pharmacogenomics Passports to Practice
STAR: Glucose control in intensive care units
ASCOPD: Medical AI models for COPD and ASTHMA inpatient risk stratification
COPowereD: Medical AI algorithms using patient reported outcomes for COPD to detect and predict hospitalization or acute exacerbations
Current Implementation
The current version (Proof of Concept) focuses on Use Case 4: ASCOPD - Medical AI models for COPD and ASTHMA inpatient risk stratification.

Quality Dimensions
The framework evaluates data quality across the following dimensions:

Population Representativity: Evaluates class balance
Metadata Granularity: Checks availability of metadata
Accuracy: Verifies if data falls within expected ranges
Coherence: Ensures consistent data types across features
Semantic Coherence: Detects columns with different names but same values, or same names but different values
Completeness: Assesses percentage of missing values
Relational Consistency: Identifies duplicate rows
Installation
bash
git clone https://github.com/your-username/Standard_Data_Quality_Framework.git
cd Standard_Data_Quality_Framework
pip install -r requirements.txt
Usage
Run the Streamlit application:

bash
streamlit run app.py
Then:

Select a use case from the dropdown
Upload your dataset (CSV format)
Optionally upload metadata
View the quality assessment results
Example Data
The repository includes example datasets for each use case in the assets folder.

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

