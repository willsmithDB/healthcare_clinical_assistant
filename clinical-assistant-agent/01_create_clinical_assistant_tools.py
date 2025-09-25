# Databricks notebook source
# MAGIC %md
# MAGIC # Building Agent Systems with Databricks -- HealthVerity Clinical Assistant Demo
# MAGIC
# MAGIC ## Part 1 - Architect Your First Agent
# MAGIC This agent will follow the workflow of a healthcare provider analyzing patient data to illustrate various agent capabilities using HealthVerity's real-world healthcare dataset. 
# MAGIC We'll assume the provider would be trying to quickly understand a patient's healthcare journey across medical claims, pharmacy data, and procedures to make informed clinical decisions.
# MAGIC
# MAGIC ### 1.1 Build Simple Tools
# MAGIC - **SQL Functions**: Create queries that access HealthVerity's real-world healthcare data including medical claims, pharmacy claims, diagnoses, and procedures.
# MAGIC - **Simple Python Function**: Create and register a Python function to overcome some common limitations of language models when analyzing healthcare data.
# MAGIC
# MAGIC ### 1.2 Integrate with an LLM [AI Playground]
# MAGIC - Combine the tools you created with a Language Model (LLM) in the AI Playground.
# MAGIC
# MAGIC ### 1.3 Test the Agent [AI Playground]
# MAGIC - Ask the agent a question and observe the response.
# MAGIC - Dive deeper into the agent’s performance by exploring MLflow traces.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -qqqq -r requirements.txt
# MAGIC # Restart to load the packages into the Python environment
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

source_catalog_name = config["source_catalog_name"]
source_schema_name = config["source_schema_name"]
target_catalog_name = config["target_catalog_name"]
target_schema_name = config["target_schema_name"]
service_date = config["service_date"]
claim_id = config["claim_id"]
patient_id = config["patient_id"]
diagnosis_code = config["diagnosis_code"]
ndc_code = config["ndc_code"]
model_uc_name = config["model_uc_name"]
alias = config["alias"]
endpoint_name = config["endpoint_name"]

# COMMAND ----------

# DBTITLE 1,Parameter Configs
from databricks.sdk import WorkspaceClient
import yaml
import os

w = WorkspaceClient()
workspace_id = str(w.get_workspace_id())

print(f"Source data from: {source_catalog_name}.{source_schema_name}")
print(f"Target functions will be created in: {target_catalog_name}.{target_schema_name}")

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

# Allows us to reference these values directly in the SQL/Python function creation
dbutils.widgets.text("source_catalog_name", defaultValue=source_catalog_name)
dbutils.widgets.text("source_schema_name", defaultValue=source_schema_name)
dbutils.widgets.text("target_catalog_name", defaultValue=target_catalog_name)
dbutils.widgets.text("target_schema_name", defaultValue=target_schema_name)
dbutils.widgets.text("workspace_id", defaultValue=workspace_id)
dbutils.widgets.text("service_date", defaultValue=service_date)

dbutils.widgets.text("claim_id", defaultValue=claim_id)
dbutils.widgets.text("patient_id", defaultValue=patient_id)
dbutils.widgets.text("diagnosis_code", defaultValue=diagnosis_code)
dbutils.widgets.text("ndc_code", defaultValue=ndc_code)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_catalog_name}.{target_schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # HealthVerity Clinical Assistant Workflow
# MAGIC
# MAGIC Below is a structured outline of the **key steps** a healthcare provider might follow when **analyzing patient data using HealthVerity's real-world healthcare dataset**. 
# MAGIC ---
# MAGIC
# MAGIC ## 1. Get patient enrollment and demographics
# MAGIC - **Action**: Retrieve patient enrollment information and demographics from the enrollment table
# MAGIC - **Why**: Understanding patient demographics, enrollment periods, and benefit types provides essential context for healthcare analysis.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Get patient enrollment information
# MAGIC %sql
# MAGIC -- find patient enrollment information given a patient id
# MAGIC select * from IDENTIFIER(:source_catalog_name || '.' || :source_schema_name || '.' || 'enrollment')
# MAGIC where patient_id = :patient_id
# MAGIC order by date_start desc
# MAGIC limit 1;

# COMMAND ----------

# DBTITLE 1,Create a function registered to Unity Catalog for getting patient enrollment
# MAGIC %sql
# MAGIC -- First lets make sure it doesn't already exist
# MAGIC DROP FUNCTION IF EXISTS IDENTIFIER(:target_catalog_name || '.' || :target_schema_name || '.' || 'get_patient_enrollment');
# MAGIC
# MAGIC -- Now we create our first function. This takes in patient id and returns enrollment information.
# MAGIC CREATE OR REPLACE FUNCTION ${target_catalog_name}.${target_schema_name}.get_patient_enrollment(
# MAGIC   patient_id STRING COMMENT 'The patient id'
# MAGIC   )
# MAGIC RETURNS TABLE (
# MAGIC   patient_id STRING, 
# MAGIC   patient_gender STRING, 
# MAGIC   patient_year_of_birth STRING, 
# MAGIC   patient_zip3 STRING, 
# MAGIC   patient_state STRING, 
# MAGIC   date_start DATE, 
# MAGIC   date_end DATE,
# MAGIC   benefit_type STRING,
# MAGIC   pay_type STRING
# MAGIC )
# MAGIC COMMENT 'Returns patient enrollment information including demographics, enrollment dates, and benefit details for a particular patient given their unique patient ID.'
# MAGIC RETURN
# MAGIC (
# MAGIC   SELECT 
# MAGIC     patient_id, 
# MAGIC     patient_gender, 
# MAGIC     patient_year_of_birth, 
# MAGIC     patient_zip3, 
# MAGIC     patient_state, 
# MAGIC     date_start, 
# MAGIC     date_end,
# MAGIC     benefit_type,
# MAGIC     pay_type
# MAGIC   FROM ${source_catalog_name}.${source_schema_name}.enrollment
# MAGIC   WHERE patient_id = patient_id
# MAGIC   ORDER BY date_start DESC
# MAGIC   LIMIT 1
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Test function call to retrieve patient enrollment
# MAGIC %sql
# MAGIC select * from ${target_catalog_name}.${target_schema_name}.get_patient_enrollment(:patient_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 2. Retrieve Medical Claims
# MAGIC - **Action**: Access medical claims data for a patient on a specific service date
# MAGIC - **Why**: Medical claims provide detailed information about healthcare services, procedures, and costs for comprehensive patient analysis.
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Create function to retrieve medical claims
# MAGIC %sql
# MAGIC -- First lets make sure it doesn't already exist
# MAGIC DROP FUNCTION IF EXISTS IDENTIFIER(:target_catalog_name || '.' || :target_schema_name || '.' || 'get_medical_claims');
# MAGIC
# MAGIC -- Now we create our second function. This takes in patient_id and service_date
# MAGIC CREATE OR REPLACE FUNCTION ${target_catalog_name}.${target_schema_name}.get_medical_claims(
# MAGIC   patient_input_id STRING COMMENT 'The patient id', 
# MAGIC   service_date DATE COMMENT 'The date of service'
# MAGIC   )
# MAGIC RETURNS TABLE (
# MAGIC   claim_id STRING, 
# MAGIC   patient_id STRING, 
# MAGIC   date_service DATE, 
# MAGIC   date_service_end DATE,
# MAGIC   location_of_care STRING,
# MAGIC   pay_type STRING
# MAGIC )
# MAGIC COMMENT 'Returns medical claims information including claim id, patient id, service dates, location of care, and payment type for a particular patient on a specific service date'
# MAGIC RETURN
# MAGIC (
# MAGIC   SELECT 
# MAGIC     claim_id, 
# MAGIC     patient_id, 
# MAGIC     date_service, 
# MAGIC     date_service_end,
# MAGIC     location_of_care,
# MAGIC     pay_type
# MAGIC   FROM ${source_catalog_name}.${source_schema_name}.medical_claim
# MAGIC   WHERE patient_id = patient_input_id and date_service = service_date
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Test function
# MAGIC %sql
# MAGIC select * from ${target_catalog_name}.${target_schema_name}.get_medical_claims(:patient_id, :service_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 3. Make Functions for Diagnosis Codes and Pharmacy Claims
# MAGIC - **Action**: get information about diagnosis codes for a patient and retrieve pharmacy claims data
# MAGIC - **Why**: Healthcare providers need to understand patient conditions through diagnosis codes and medication history through pharmacy claims for comprehensive care analysis

# COMMAND ----------

# DBTITLE 1,test sql query for patient diagnoses
# MAGIC %sql
# MAGIC -- first one gets diagnosis information for a specific patient
# MAGIC select patient_id, claim_id, date_service, diagnosis_code, diagnosis_qual, admit_diagnosis_ind
# MAGIC from IDENTIFIER(:source_catalog_name || '.' || :source_schema_name || '.' || 'diagnosis')
# MAGIC where patient_id = :patient_id;

# COMMAND ----------

# DBTITLE 1,test sql query for pharmacy claims
# MAGIC %sql
# MAGIC -- second one returns pharmacy claims for a given patient
# MAGIC select claim_id, patient_id, date_service, ndc, fill_number, days_supply, dispensed_quantity, pay_type, copay_coinsurance
# MAGIC from IDENTIFIER(:source_catalog_name || '.' || :source_schema_name || '.' || 'pharmacy_claim')
# MAGIC where patient_id = :patient_id;

# COMMAND ----------

# DBTITLE 1,make function for patient diagnoses
# MAGIC %sql
# MAGIC -- First lets make sure it doesn't already exist
# MAGIC DROP FUNCTION IF EXISTS IDENTIFIER(:target_catalog_name || '.' || :target_schema_name || '.' || 'get_patient_diagnoses');
# MAGIC
# MAGIC -- This takes in patient_id to get all diagnosis codes for that patient
# MAGIC CREATE OR REPLACE FUNCTION ${target_catalog_name}.${target_schema_name}.get_patient_diagnoses(
# MAGIC   patient_input_id STRING COMMENT 'The patient id'
# MAGIC   )
# MAGIC RETURNS TABLE (
# MAGIC   claim_id STRING, 
# MAGIC   patient_id STRING, 
# MAGIC   date_service DATE, 
# MAGIC   date_service_end DATE,
# MAGIC   diagnosis_code STRING,
# MAGIC   diagnosis_qual STRING,
# MAGIC   admit_diagnosis_ind STRING
# MAGIC )
# MAGIC COMMENT 'Returns diagnosis information including claim id, patient id, service dates, diagnosis code, qualifier, and admission diagnosis indicator for a specific patient'
# MAGIC RETURN
# MAGIC (
# MAGIC   SELECT 
# MAGIC     claim_id, 
# MAGIC     patient_id, 
# MAGIC     date_service, 
# MAGIC     date_service_end,
# MAGIC     diagnosis_code,
# MAGIC     diagnosis_qual,
# MAGIC     admit_diagnosis_ind
# MAGIC   FROM ${source_catalog_name}.${source_schema_name}.diagnosis
# MAGIC   WHERE patient_id = patient_input_id
# MAGIC   ORDER BY date_service DESC
# MAGIC   LIMIT 50
# MAGIC );

# COMMAND ----------

# DBTITLE 1,test func for getting patient diagnoses
# MAGIC %sql
# MAGIC select * from ${target_catalog_name}.${target_schema_name}.get_patient_diagnoses(:patient_id)

# COMMAND ----------

# DBTITLE 1,make function for getting pharmacy claims
# MAGIC %sql
# MAGIC
# MAGIC -- First lets make sure it doesn't already exist
# MAGIC DROP FUNCTION IF EXISTS IDENTIFIER(:target_catalog_name || '.' || :target_schema_name || '.' || 'get_pharmacy_claims');
# MAGIC
# MAGIC -- This takes in patient_id to get pharmacy claims
# MAGIC CREATE OR REPLACE FUNCTION ${target_catalog_name}.${target_schema_name}.get_pharmacy_claims(
# MAGIC   patient_input_id STRING COMMENT 'The patient id'
# MAGIC   )
# MAGIC RETURNS TABLE (
# MAGIC   claim_id STRING, 
# MAGIC   patient_id STRING,  
# MAGIC   date_service DATE,
# MAGIC   ndc STRING,
# MAGIC   fill_number INT,
# MAGIC   days_supply INT,
# MAGIC   dispensed_quantity FLOAT,
# MAGIC   pay_type STRING,
# MAGIC   copay_coinsurance FLOAT,
# MAGIC   submitted_gross_due FLOAT,
# MAGIC   paid_gross_due FLOAT
# MAGIC )
# MAGIC COMMENT 'Returns pharmacy claims information including claim details, NDC codes, fill information, and payment details for a given patient. Example: claim_id: 66314b1117ffbdbbbbee693a9f69453d, ndc: 65162027250, days_supply: 30, dispensed_quantity: 1.0'
# MAGIC RETURN
# MAGIC (
# MAGIC   SELECT 
# MAGIC     claim_id, 
# MAGIC     patient_id,  
# MAGIC     date_service,
# MAGIC     ndc,
# MAGIC     fill_number,
# MAGIC     days_supply,
# MAGIC     dispensed_quantity,
# MAGIC     pay_type,
# MAGIC     copay_coinsurance,
# MAGIC     submitted_gross_due,
# MAGIC     paid_gross_due
# MAGIC   FROM ${source_catalog_name}.${source_schema_name}.pharmacy_claim
# MAGIC   WHERE patient_id = patient_input_id
# MAGIC   ORDER BY date_service DESC 
# MAGIC   LIMIT 50
# MAGIC );

# COMMAND ----------

# DBTITLE 1,test function for pharmacy claims
# MAGIC %sql
# MAGIC select * from ${target_catalog_name}.${target_schema_name}.get_pharmacy_claims(:patient_id)
# MAGIC
# MAGIC -- this will return all pharmacy claims for the patient including medication details and costs 

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 4. Create Function for Procedure Analysis
# MAGIC - **Action**: Retrieve procedure codes and details for comprehensive healthcare analysis
# MAGIC - **Why**: Understanding procedures performed helps complete the patient's healthcare story alongside diagnoses and medications
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,make function for getting procedure information
# MAGIC %sql
# MAGIC
# MAGIC -- First lets make sure it doesn't already exist
# MAGIC DROP FUNCTION IF EXISTS IDENTIFIER(:target_catalog_name || '.' || :target_schema_name || '.' || 'get_patient_procedures');
# MAGIC
# MAGIC -- This takes in patient_id to get procedure information
# MAGIC CREATE OR REPLACE FUNCTION ${target_catalog_name}.${target_schema_name}.get_patient_procedures(
# MAGIC   patient_input_id STRING COMMENT 'The patient id'
# MAGIC   )
# MAGIC RETURNS TABLE (
# MAGIC   claim_id STRING, 
# MAGIC   patient_id STRING,
# MAGIC   service_line_number STRING,
# MAGIC   date_service DATE,
# MAGIC   date_service_end DATE,
# MAGIC   procedure_code STRING,
# MAGIC   procedure_qual STRING,
# MAGIC   procedure_units FLOAT,
# MAGIC   revenue_code STRING,
# MAGIC   line_charge FLOAT,
# MAGIC   line_allowed FLOAT
# MAGIC )
# MAGIC COMMENT 'Returns procedure information including claim details, procedure codes, service dates, units, and charges for a given patient'
# MAGIC RETURN
# MAGIC (
# MAGIC   SELECT 
# MAGIC     claim_id, 
# MAGIC     patient_id,
# MAGIC     service_line_number,
# MAGIC     date_service,
# MAGIC     date_service_end,
# MAGIC     procedure_code,
# MAGIC     procedure_qual,
# MAGIC     procedure_units,
# MAGIC     revenue_code,
# MAGIC     line_charge,
# MAGIC     line_allowed
# MAGIC   FROM ${source_catalog_name}.${source_schema_name}.procedure
# MAGIC   WHERE patient_id = patient_input_id
# MAGIC   ORDER BY date_service DESC 
# MAGIC   LIMIT 50
# MAGIC );

# COMMAND ----------

# DBTITLE 1,test function for procedures
# MAGIC %sql
# MAGIC select * from ${target_catalog_name}.${target_schema_name}.get_patient_procedures(:patient_id)

# COMMAND ----------

# DBTITLE 1,Let's take a look at our created functions
from IPython.display import display, HTML

# Retrieve the Databricks host URL
workspace_url = spark.conf.get('spark.databricks.workspaceUrl')

# Create HTML link to created functions
html_link = f'<a href="https://{workspace_url}/explore/data/functions/{target_catalog_name}/{target_schema_name}/get_patient_enrollment" target="_blank">Go to Unity Catalog to see Registered Functions</a>'
display(HTML(html_link))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Now lets go over to the AI Playground to see how we can use these functions and assemble our first HealthVerity Agent!
# MAGIC
# MAGIC ### The AI Playground can be found on the left navigation bar under 'Machine Learning' or you can use the link created below
# MAGIC
# MAGIC Example questions to try:
# MAGIC - `What is the healthcare journey for patient 4baf3314e4a181c5effcf2751fbe1e21?`
# MAGIC - `What medications has patient 4baf3314e4a181c5effcf2751fbe1e21 been prescribed?`
# MAGIC - `What are the diagnosis codes for patient 4baf3314e4a181c5effcf2751fbe1e21?`

# COMMAND ----------

# DBTITLE 1,Create link to AI Playground
# Create HTML link to AI Playground
html_link = f'<a href="https://{workspace_url}/ml/playground" target="_blank">Go to AI Playground</a>'
display(HTML(html_link))

# COMMAND ----------


