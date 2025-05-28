# Standard Data Quality Framework (SDQF)
A comprehensive Streamlit-based web application for assessing data quality across multiple dimensions using both qualitative and quantitative metrics.
## 🚀 Features

Dual Assessment Approach: Combines qualitative expert judgment with quantitative automated checks
Interactive Web Interface: User-friendly Streamlit application with visual feedback
Multiple Quality Dimensions: Evaluates 14 different aspects of data quality
Visual Analytics: Radar charts and detailed metrics for comprehensive assessment
Flexible Input Options: Manual configuration or JSON upload for assessments
Extensible Framework: Support for multiple use cases (currently ASCOPD implemented)

## 📊 Quality Dimensions
### Quantitative Metrics (Automated)

Population Representativity: Measures class balance in target variable
Metadata Granularity: Ratio of records with metadata
Accuracy: Proportion of values within expected ranges
Coherence: Consistency of data types across features
Semantic Coherence: Uniqueness of column names
Completeness: Proportion of non-missing values
Relational Consistency: Uniqueness of data rows

### Qualitative Metrics (Expert Assessment)

Accessibility: Ease of obtaining and understanding data
Use Permissiveness: License and governance flexibility
Availability: Data presence and readiness for use
Compliance: Adherence to standards and regulations
Provenance: Documentation of data source and generation
Trustworthiness: Trust indicators in documentation
Consistency: Logical coherence and freedom from contradictions

## 🛠️ Installation
### Prerequisites

Python 3.8 or higher
pip package manager

### Setup

1. Clone the repository:

bashgit clone https://github.com/yourusername/standard-data-quality-framework.git
cd standard-data-quality-framework

2. Create a virtual environment:

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

bashpip install -r requirements.txt

## 🚦 Getting Started

1. Run the Streamlit application:

bashstreamlit run app.py

2. Open your browser to http://localhost:8501
3. Follow the workflow:

- Select a use case 
- Upload your dataset
- Optionally upload metadata CSV
- Click "Load Data"
- Complete the qualitative assessment questionnaire
- Configure quantitative checks (select target, age, and height columns)
- Click "Run Quality Checks"
- Review the results


## 📈 Scoring System
All metrics use a consistent 1-5 rating scale:

- 1/5: Poor quality (0-20% score)
- 2/5: Below average (20-40% score)
- 3/5: Average (40-60% score)
- 4/5: Good (60-80% score)
- 5/5: Excellent (80-100% score)


## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
## 👥 Authors

MetaMind Innovations PC