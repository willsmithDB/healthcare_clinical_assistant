# Databricks HLS Clinical Assistant

A comprehensive healthcare AI assistant built on Databricks using HealthVerity's real-world healthcare dataset. This project demonstrates how to create, deploy, evaluate, and monitor clinical AI agents that can help healthcare professionals analyze patient healthcare journeys including medical claims, pharmacy data, diagnoses, and procedures.

## DISCLAIMER - this code is for reference and not an official Databricks asset. There are no assurances nor guarantees on this for working in any environment. Use for educational purposes only. See LICENSE for more details. 

## 🏥 Overview

The Clinical Assistant is designed to follow the workflow of a healthcare provider analyzing patient data to make informed clinical decisions. It enables healthcare professionals to quickly understand a patient's healthcare journey across medical claims, pharmacy data, diagnoses, and procedures using HealthVerity's real-world healthcare dataset.

### Key Capabilities

- **Patient Enrollment Analysis**: Retrieve patient demographics, enrollment periods, and benefit information
- **Medical Claims Analysis**: Access detailed medical claims data including procedures and healthcare services
- **Pharmacy Claims Tracking**: Analyze prescription history, NDC codes, and medication dispensing patterns
- **Diagnosis Code Analysis**: Extract and analyze patient diagnosis codes and medical conditions
- **Procedure Analysis**: Review healthcare procedures, codes, and associated costs
- **Conversational Interface**: Natural language interaction powered by large language models
- **Comprehensive Monitoring**: Built-in evaluation and monitoring framework for safety and accuracy

## 🏗️ Architecture

### Components

1. **HealthVerity Clinical Tools** (`01_create_clinical_assistant_tools.py`)
   - SQL functions for accessing HealthVerity's real-world healthcare dataset
   - Unity Catalog function registration for patient enrollment, medical claims, pharmacy claims, diagnoses, and procedures
   - Data retrieval and analysis tools for comprehensive healthcare data
   - Configuration-driven parameter management

2. **Agent Implementation** (`clinical-assistant-agent/`)
   - **Primary Agent** (`agent.py`): Claude 3.7 Sonnet-powered conversational agent
   - **Base Agent** (`base_agent.py`): Alternative configuration with enhanced prompt engineering
   - **Configuration Management** (`config.yaml`): Centralized parameter control
   - LangChain/LangGraph-based architecture with MLflow integration
   - Tool orchestration and response generation

3. **Evaluation Framework** (`03_evaluations.ipynb`)
   - Automated evaluation data generation
   - Performance metrics and benchmarking
   - Quality assessment workflows

4. **Monitoring System** (`04_monitoring.ipynb`)
   - Real-time response monitoring
   - Safety and groundedness assessments
   - Custom guidelines enforcement

## 🚀 Quick Start

### Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to HealthVerity dataset (`HealthVerity_Claims_Sample_Patient_Dataset.hv_claims_sample` catalog)
   - You can create a read-only catalog from the Databricks Marketplace under Claims Sample Patient Dataset
- MLflow and AI Agent Framework permissions

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   cd clinical-assistant-agent
   pip install -r requirements.txt
   ```

2. **Configure the Environment**
   - **IMPORTANT**: Update `clinical-assistant-agent/config.yaml` with your workspace settings before using the system:
     - Catalog and schema names
     - Model names and endpoints
     - Default patient/admission IDs for testing
     - Experiment ID for MLflow tracking
     - User email addresses for monitoring

3. **Set Up Clinical Tools**
   - Run `clinical-assistant-agent/01_create_clinical_assistant_tools.py` to create SQL functions
   - This registers functions in Unity Catalog for data access

4. **Deploy the Agent**
   - Run `clinical-assistant-agent/02_clinical-assistant-driver.py` to:
     - Log the agent to MLflow with automatic tracing
     - Register the model in Unity Catalog with "Champion" alias
     - Deploy to serving endpoints (optional)
     - Set up monitoring with custom judges
   
5. **Set Up Evaluation and Monitoring**
   - Run `clinical-assistant-agent/03_evaluations.ipynb` to create evaluation datasets
   - Run `clinical-assistant-agent/04_monitoring.ipynb` to configure real-time monitoring

## 🔧 HealthVerity Clinical Tools

The system includes five main SQL functions registered in Unity Catalog for comprehensive healthcare data analysis:

### `get_patient_enrollment(patient_id)`
Returns patient demographics and enrollment information including:
- Patient gender, year of birth, ZIP3, and state
- Enrollment start and end dates
- Benefit type and payment type information

### `get_medical_claims(patient_id, service_date)`
Retrieves medical claims for a patient on a specific service date:
- Claim ID, patient ID, service dates
- Location of care and payment type
- Healthcare service details

### `get_patient_diagnoses(patient_id)`
Returns comprehensive diagnosis information for a patient:
- Diagnosis codes and qualifiers
- Service dates and admission diagnosis indicators
- Medical condition tracking over time

### `get_pharmacy_claims(patient_id)`
Provides pharmacy claims and medication history:
- NDC codes for medication identification
- Fill numbers, days supply, and dispensed quantities
- Payment details including copays and gross amounts

### `get_patient_procedures(patient_id)`
Extracts procedure information and healthcare services:
- Procedure codes and qualifiers
- Service dates and procedure units
- Revenue codes and associated charges

## ⚙️ Technical Specifications

### Model Endpoints
- **Primary LLM**: `databricks-claude-3-7-sonnet`
- **Alternative LLM**: `databricks-meta-llama-3-3-70b-instruct`
- **Deployment**: Model serving endpoints with Unity Catalog registration

### Configuration Parameters
The system uses `config.yaml` for centralized configuration. **Note: These variables must be updated with your actual workspace values before use:**
```yaml
# Source data (HealthVerity dataset)
source_catalog_name: "HealthVerity_Claims_Sample_Patient_Dataset"
source_schema_name: "hv_claims_sample"

# Target UC functions
target_catalog_name: "CATALOG"  # Update with your catalog
target_schema_name: "SCHEMA"    # Update with your schema

# Sample data parameters
service_date: "2021-12-21"
claim_id: "12390dfb568442cb957d5b3cfefe1119"
patient_id: "4baf3314e4a181c5effcf2751fbe1e21"
diagnosis_code: "Z79899"
ndc_code: "65162027250"

# Model configuration
model_uc_name: "CATALOG.SCHEMA.healthverity_clinical_assistant_dev"  # Update with your catalog/schema
alias: "Champion"
endpoint_name: "clinical_assistant_healthverity"

# MLflow tracking
experiment_id: "CHANGE_ME"  # Update with your experiment ID
label_users: ["user@example.com"]  # Update with actual user emails
```

### Agent Variants
- **Standard Agent**: Basic healthcare data analysis functionality
- **Enhanced Agent**: Advanced real-world claims analysis with comprehensive healthcare journey tracking

## 🤖 Agent Usage

### Basic Queries

```python
# Example questions the agent can answer:
"What is the healthcare journey for patient 4baf3314e4a181c5effcf2751fbe1e21?"
"What medications has patient 4baf3314e4a181c5effcf2751fbe1e21 been prescribed?"
"What are the diagnosis codes for patient 4baf3314e4a181c5effcf2751fbe1e21?"
"Show me the medical claims for patient 4baf3314e4a181c5effcf2751fbe1e21 on 2021-12-21"
"What procedures has patient 4baf3314e4a181c5effcf2751fbe1e21 had performed?"
"What is the enrollment information for patient 4baf3314e4a181c5effcf2751fbe1e21?"
```

### Integration Points

- **AI Playground**: Interactive testing and development
- **Genie Rooms**: Collaborative workspace with optimized Text-2-SQL capabilities
- **Agent Bricks**: Key Information Extraction (KIE) from clinical notes
- **MLflow**: Experiment tracking and model management
- **Unity Catalog**: Centralized function and data governance
- **Model Serving**: Production deployment with endpoint management

## 📊 Evaluation & Monitoring

### Evaluation Framework
- **Automated Test Generation**: Creates realistic clinical scenarios
- **Performance Metrics**: Accuracy, relevance, safety assessments
- **Benchmark Datasets**: Standardized evaluation sets

### Monitoring Capabilities
- **Real-time Assessment**: Continuous monitoring of agent responses
- **Safety Checks**: Built-in safety and bias detection
- **Quality Metrics**: Groundedness, relevance, clarity measurements
- **Custom Guidelines**: Configurable assessment criteria

### Assessment Criteria
- **Safety**: Ensures medical information is safe and appropriate
- **Groundedness**: Verifies responses are based on available data
- **Relevance to Query**: Confirms answers address the user's question
- **Chunk Relevance**: Validates information retrieval accuracy
- **Guideline Adherence**: Custom rules including:
  - English language responses
  - Clear, coherent, and concise communication
  - Relevant responses (including appropriate refusals)
  - No speculation when documentation is unavailable

## 🔒 Security & Compliance

- **Data Privacy**: All patient data is de-identified (HealthVerity dataset)
- **Access Control**: Unity Catalog-based permissions and secure data access
- **Audit Trail**: Complete MLflow tracking of all interactions
- **Safety Monitoring**: Continuous safety and bias assessment
- **Real-World Data Governance**: Secure handling of healthcare claims and pharmacy data

## 📁 Project Structure

```
dbx_hls_clinical_assistant/
├── clinical-assistant-agent/
│   ├── 01_create_clinical_assistant_tools.py # SQL function creation and setup
│   ├── 02_clinical-assistant-driver.py       # Main agent deployment workflow
│   ├── 03_evaluations.ipynb                  # Evaluation framework and testing
│   ├── 04_monitoring.ipynb                   # Real-time monitoring setup
│   ├── agent.py                              # Primary agent implementation
│   ├── base_agent.py                         # Alternative agent configuration
│   ├── config.yaml                           # Configuration parameters
│   └── requirements.txt                      # Agent-specific dependencies
├── requirements.txt                          # Project-wide dependencies
└── README.md                                # This documentation
```

## 🎯 Use Cases

### Clinical Decision Support
- Quickly review patient healthcare journey and enrollment history
- Analyze prescription patterns and medication adherence
- Understand diagnosis trends and medical conditions over time

### Healthcare Analytics
- Analyze patient populations across medical and pharmacy claims
- Study healthcare utilization patterns and costs
- Generate insights from real-world evidence data

### Healthcare Operations
- Streamline patient data review for care coordination
- Improve understanding of patient healthcare journeys
- Support value-based care initiatives with comprehensive claims analysis

## 🔗 Resources

- **[Databricks Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/build-genai-apps.html)**: Documentation
- **[HealthVerity](https://www.healthverity.com/)**: Real-world healthcare data platform

## 🤝 Contributing

This repository is for demonstration purposes. For questions or contributions, please contact the development team.

## ⚠️ Disclaimer

This HealthVerity clinical assistant is for demonstration and research purposes only. It should not be used for actual clinical decision-making without proper validation and approval from healthcare professionals and regulatory bodies. The system processes de-identified real-world healthcare data and should be used in compliance with all applicable healthcare data privacy regulations.
