# Databricks notebook source
# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

service_date = config["service_date"]
claim_id = config["claim_id"]
patient_id = config["patient_id"]
diagnosis_code = config["diagnosis_code"]
ndc_code = config["ndc_code"]
endpoint_name =  "agents_users-will_smith-" + config["endpoint_name"]

# COMMAND ----------

import mlflow
mlflow.set_experiment(experiment_id="CHANGE_ME")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Online monitoring uses live scorers now: [docs](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/production-quality-monitoring)
# MAGIC
# MAGIC ### The older monitors are now legacy: [docs](https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/concepts/production-monitoring).

# COMMAND ----------

# MAGIC %md
# MAGIC Note: At any given time, at most 20 scorers can be associated with an experiment for continuous quality monitoring.

# COMMAND ----------

# DBTITLE 1,Use a pre-defined scorer
from mlflow.genai.scorers import Safety, scorer, ScorerSamplingConfig

# Register the scorer with a name and start monitoring
safety_scorer = Safety()
safety_scorer.register(name="safety_check")
safety_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))

# Use a specific model
# safety_scorer = Safety(model="databricks:/databricks-gpt-oss-20b").register(name="safety_check")

# COMMAND ----------

# DBTITLE 1,Use a scorer with guidelines
from mlflow.genai.scorers import Guidelines

# Create and register the guidelines scorer
english_scorer = Guidelines(
  name="english",
  guidelines=["The response must be in English"]
).register(name="is_english")  # name must be unique to experiment

# Start monitoring with the specified sample rate
english_scorer = english_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.7))

# COMMAND ----------

# DBTITLE 1,List scorers
from mlflow.genai.scorers import list_scorers

# List all registered scorers
scorers = list_scorers()
for scorer in scorers:
    print(f"Name: {scorer._server_name}")
    print(f"Sample rate: {scorer.sample_rate}")
    print(f"Filter: {scorer.filter_string}")
    print("---")

# COMMAND ----------

# DBTITLE 1,Update a scorer with higher sampling rate
from mlflow.genai.scorers import get_scorer

# Get existing scorer and update its configuration (immutable operation)
safety_scorer = get_scorer(name="safety_monitor")
updated_scorer = safety_scorer.update(sampling_config=ScorerSamplingConfig(sample_rate=0.8))  # Increased from 0.5

# Note: The original scorer remains unchanged; update() returns a new scorer instance
print(f"Original sample rate: {safety_scorer.sample_rate}")  # Original rate
print(f"Updated sample rate: {updated_scorer.sample_rate}")   # New rate

# COMMAND ----------

# MAGIC %md
# MAGIC # Test Traffic

# COMMAND ----------

from mlflow import deployments

client = deployments.get_deploy_client("databricks")

questions = [
    "What information is there for the claim id 12390dfb568442cb957d5b3cfefe1119?",
    "What is the medical journey for patient: 4baf3314e4a181c5effcf2751fbe1e21",
    "What claims has the patient 4baf3314e4a181c5effcf2751fbe1e21 had?",
    "Which claims are associated with the ndc_code: 65162027250?",
    "Which claims are associated with the diagnosis_code: Z79899?",
    "What information is there for patient_id: 4baf3314e4a181c5effcf2751fbe1e21 for the service_date: \"2021-12-21\"",
]

for i, question in enumerate(questions, 1):
    print(f"\nQuestion {i}: {question}")  
    response = client.predict(
        endpoint=endpoint_name,
        inputs={
            "messages": [
                {"role": "user", "content": question}
            ]
        }
    )
    print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC # To see monitoring results:
# MAGIC After scheduling scorers, allow 15-20 minutes for initial processing. Then:
# MAGIC - Navigate to your MLflow experiment.
# MAGIC - Open the Traces tab to see assessments attached to traces.
# MAGIC - Use the monitoring dashboards to track quality trends.
