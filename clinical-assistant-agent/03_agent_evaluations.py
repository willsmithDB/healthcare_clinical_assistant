# Databricks notebook source
# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC
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
experiment_path = config['experiment_path']
label_users = config["label_users"]

# COMMAND ----------

# MAGIC %md
# MAGIC # Load back the model and experiment

# COMMAND ----------

import mlflow 
experiment_info = mlflow.set_experiment(experiment_path)

# COMMAND ----------

experiment_info

# COMMAND ----------

import mlflow
client = mlflow.MlflowClient()

client.get_model_version_by_alias(model_uc_name, alias)

# COMMAND ----------

model_version_uri = f"models:/{model_uc_name}@{alias}"
model = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

model.predict({"messages": [{"role": "user", "content": f"What enrollment information do you have for patient {patient_id}?"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the agent with [Agent Evaluation](https://docs.databricks.com/generative-ai/agent-evaluation/index.html)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC To evaluate your tool calls and custom metrics, try adding [tool call metrics](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/custom-metrics#evaluating-tool-calls) and [custom metrics](https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/custom-metrics#develop-custom-metrics).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Since we dont have any eval dataset yet and we are too lazy to write our own, e.g., even 10 rows, we are gonna synthesize the eval dataset leveraging the `generate_evals_df` function.

# COMMAND ----------

# MAGIC %md
# MAGIC __Note:__ We need to provide `docs` argument to the `generate_evals_df` function later as context for generation. For `docs`, it requires two fields:
# MAGIC - content
# MAGIC - doc_uri
# MAGIC
# MAGIC ref: 
# MAGIC 1. https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/synthesize-evaluation-set
# MAGIC 2. https://docs.databricks.com/aws/en/generative-ai/agent-evaluation/evaluation-set

# COMMAND ----------

# DBTITLE 1,Generate the docs
docs = spark.sql(
    f"""
    SELECT
        patient_id as id, 
        CONCAT_WS(
            ' ', 
            'PATIENT_ID:', patient_id, 
            'CLAIM_ID:', claim_id, 
            'DATE_SERVICE:', CAST(date_service AS STRING), 
            'LOCATION_OF_CARE:', location_of_care, 
            'PAY_TYPE:', pay_type
        ) as content, 
        CONCAT_WS('_', 'PATIENT_ID', patient_id, 'CLAIM_ID', claim_id) as doc_uri
    FROM {source_catalog_name}.{source_schema_name}.medical_claim
    WHERE patient_id = '{patient_id}'
    ORDER BY date_service DESC
    LIMIT 1000
    """
)

docs = docs.distinct()

display(docs)

# COMMAND ----------

# MAGIC %md
# MAGIC **A Hint:**
# MAGIC If you are lazy and you dont want to manually type the 'agent_description' and 'question_guidelines' content below. Consider utilize the Databricks Assistant on your right panel to write for you. Just copy & paste.
# MAGIC
# MAGIC Now it already has content filled in. You can proceed like usual.

# COMMAND ----------

# DBTITLE 1,Synthesize Eval dataset
from databricks.agents.evals import generate_evals_df

eval_data = generate_evals_df(
  docs, 
  num_evals=10,   
  agent_description="An agent that retrieves and analyzes patient healthcare data from HealthVerity's real-world healthcare dataset. The agent can use the following tools: get_patient_enrollment to get patient demographics and enrollment information, get_medical_claims to get medical claims for a patient on a specific service date, get_patient_diagnoses to get all diagnosis codes for a patient, get_pharmacy_claims to get pharmacy claims and medication history, and get_patient_procedures to get procedure codes and details for a patient.",
  question_guidelines=f"""
  The agent can use the following tools:
  - get_patient_enrollment: Takes a patient_id as input. Returns patient demographics, enrollment dates, and benefit details for a particular patient.
  - get_medical_claims: Takes a patient_id and service_date as input. Returns medical claims information including claim id, service dates, location of care, and payment type for a particular patient on a specific service date.
  - get_patient_diagnoses: Takes a patient_id as input. Returns diagnosis information including diagnosis codes, qualifiers, and service dates for a specific patient.
  - get_pharmacy_claims: Takes a patient_id as input. Returns pharmacy claims information including NDC codes, fill information, days supply, and payment details for a given patient.
  - get_patient_procedures: Takes a patient_id as input. Returns procedure information including procedure codes, service dates, units, and charges for a given patient.

  Any dates that are returned should be in full date format when possible. 

  Focus on extracting comprehensive healthcare information. Use the provided tools to gather detailed data about the patient's healthcare journey including medical claims, pharmacy history, diagnoses, and procedures. You must include the relevant input in the prompt so that the LLM may use the relevant tool appropriately. For example, if you ask a question about a patient, include the patient's id in the prompt: What is the healthcare journey for patient '{patient_id}'? 
  
  """
)

# COMMAND ----------

# MAGIC %md
# MAGIC __Let's take a look at the evals that we generated:__

# COMMAND ----------

display(eval_data)

# COMMAND ----------

# DBTITLE 1,eval dataset I/O
spark.createDataFrame(eval_data).write.mode("append").saveAsTable(f"{target_catalog_name}.{target_schema_name}.clinical_assistant_eval_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Scorers
# MAGIC We are now going to use mlflow 3.x for the newest features!
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC For @mlflow.trace details, ref: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/data-model

# COMMAND ----------

# DBTITLE 1,Wrapper function for predictions when evaluating
import mlflow
from mlflow.genai.scorers import scorer

# Wrap our model in predict function to handle parameter mapping 
@mlflow.trace
def evaluate_model(messages: dict) -> dict:
    return model.predict({"messages": messages})

# COMMAND ----------

# DBTITLE 1,import default scorers and Define the guidelines for our evaluation
from databricks.agents.evals import judges, metric
from mlflow.genai.scorers import (
    Correctness, RetrievalSufficiency,  
    RelevanceToQuery, Safety, RetrievalGroundedness, RetrievalRelevance, Guidelines
)

# Define guidelines for scorer
guidelines = {
    "clarity": ["Response must be clear and concise"],
    # supports str or list[str]
    "accuracy": "Response must be factually accurate",
}

# COMMAND ----------

# DBTITLE 1,Define a custom scorer for our evaluations
import mlflow
from mlflow.entities import Trace, Feedback
from mlflow.genai.judges import is_context_relevant
from mlflow.genai.scorers import scorer
from typing import Any

@scorer
def is_message_relevant(inputs: dict[str, Any], outputs: str) -> Feedback:
    # The `inputs` field for `sample_app` is a dictionary like: {"messages": [{"role": ..., "content": ...}, ...]}
    # We need to extract the content of the last user message to pass to the relevance judge.

    last_user_message_content = None
    if "messages" in inputs and isinstance(inputs["messages"], list):
        for message in reversed(inputs["messages"]):
            if message.get("role") == "user" and "content" in message:
                last_user_message_content = message["content"]
                break

    if not last_user_message_content:
        raise Exception("Could not extract the last user message from inputs to evaluate relevance.")

    # Call the `relevance_to_query judge. It will return a Feedback object.
    return is_context_relevant(
        request=last_user_message_content,
        context={"response": outputs},
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Individual Scorer

# COMMAND ----------

eval_data["inputs"][0]

# COMMAND ----------

eval_data.display()

# COMMAND ----------

from mlflow.genai import judges

result = judges.is_correct(
    request=eval_data["inputs"][0],
    response=model.predict(eval_data["inputs"][0]),
    expected_facts=eval_data["expectations"][0]["expected_facts"]
)
print(f"Judge result: {result.value}")
print(f"Rationale: {result.rationale}")

# COMMAND ----------

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=evaluate_model,
    scorers=[
        # With ground truth judges
        Correctness(),
        RetrievalSufficiency(),
        Guidelines(name="clarity", guidelines=guidelines["clarity"]),
        Guidelines(name="accuracy", guidelines=guidelines["accuracy"]),
        # Without ground truth judges
        RelevanceToQuery(),
        Safety(),
        RetrievalGroundedness(),
        RetrievalRelevance(),
        # Custom scorers
        # check_no_pii,
        is_message_relevant
    ]
)

# COMMAND ----------

results.tables['eval_results'].display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/build-eval-dataset#seeding-an-evaluation-dataset-with-synthetic-data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Built the Final Eval Dataset with the previous synthesized dataset and the evaluation traces columns above
# MAGIC
# MAGIC ref: https://learn.microsoft.com/en-us/azure/databricks/mlflow3/genai/eval-monitor/build-eval-dataset

# COMMAND ----------

import mlflow
import mlflow.genai.datasets
import time

uc_schema = f"{target_catalog_name}.{target_schema_name}"
# This table will be created in the above UC schema
evaluation_dataset_table_name = "full_evaluation_dataset"

try:
    mlflow.genai.datasets.delete_dataset(
        uc_table_name=f"{uc_schema}.{evaluation_dataset_table_name}"
    )
except Exception as e:
    print(f"Caught error: {e}")

try:
    eval_dataset = mlflow.genai.datasets.create_dataset(
        uc_table_name=f"{uc_schema}.{evaluation_dataset_table_name}",
    )
    print(f"Created evaluation dataset: {uc_schema}.{evaluation_dataset_table_name}")
except Exception as e: 
    if "TABLE_ALREADY_EXISTS" in str(e):
        print("Table already exists. Loading dataset instead.")
        eval_dataset = mlflow.genai.datasets.get_dataset(f"{uc_schema}.{evaluation_dataset_table_name}")
        print(f"Loaded evaluation dataset: {uc_schema}.{evaluation_dataset_table_name}")
    else:
        print(f"Caught error: {e}")

# COMMAND ----------

#: same as the 'result' variable above, we now fetch it using mlflow.search_traces() cause we logged the traces before. 
traces = mlflow.search_traces(
    filter_string="attributes.status = 'OK'",
    order_by=["attributes.timestamp_ms DESC"],
    max_results=5
)

print(f"Found {len(traces)} successful traces")

# COMMAND ----------

traces.display()

# COMMAND ----------

# DBTITLE 1,expect to be empty now
eval_dataset.to_df()

# COMMAND ----------

# DBTITLE 1,then we merge traces into it
eval_dataset = eval_dataset.merge_records(traces)
print(f"Added {len(traces)} records to evaluation dataset")

# COMMAND ----------

# DBTITLE 1,now we should have records from traces
eval_dataset.to_df()

# COMMAND ----------

# Preview the dataset
eval_df = eval_dataset.to_df()
print(f"\nDataset preview:")
print(f"Total records: {len(eval_df)}")
print("\nSample record:")
sample = eval_df.iloc[0]
print(f"Inputs: {sample['inputs']}")

# COMMAND ----------

# DBTITLE 1,you will find the UC table was updated as well when you merge_record
spark.table(f"{uc_schema}.{evaluation_dataset_table_name}").display()

# COMMAND ----------

# MAGIC %md
# MAGIC > __NOTE:__ So far, we have the `eval_dataset` filled with traces. Trace have AI scores inside, but no human-in-the-loop yet. Next step we are gonna provide human (SME) review/labeling to the dataset via labeling session and review app.

# COMMAND ----------

# DBTITLE 1,Skip this
# spark.sql(f"""
# CREATE OR REPLACE TABLE {target_catalog_name}.{target_schema_name}.augmented_eval_data (
#     dataset_record_id STRING,
#     create_time TIMESTAMP,
#     created_by STRING,
#     last_update_time TIMESTAMP,
#     last_updated_by STRING,
#     source STRUCT<
#         human: STRUCT<user_name: STRING>,
#         document: STRUCT<doc_uri: STRING, content: STRING>,
#         trace: STRUCT<trace_id: STRING>
#     >,
#     inputs STRING,
#     expectations STRING,
#     tags MAP<STRING, STRING>
# )
# """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Human Feedback / Labeling

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Schema and Session

# COMMAND ----------

# DBTITLE 1,Create MLflow Labeling Session
import mlflow
import mlflow.genai.labeling as labeling
import mlflow.genai.label_schemas as schemas

quality_schema = schemas.create_label_schema(
    name="response_quality",
    type=schemas.LabelSchemaType.FEEDBACK,
    title="Rate the response quality",
    input=schemas.InputCategorical(
        options=["Poor", "Fair", "Good", "Excellent"]
    ),
    overwrite=True
)

expected_facts_schema = schemas.create_label_schema(
    name=schemas.EXPECTED_FACTS,
    type=schemas.LabelSchemaType.EXPECTATION,
    title="Expected facts",
    input=schemas.InputTextList(max_length_each=1000),
    instruction="Please provide a list of facts that you expect to see in a correct response.",
    overwrite=True
)

# Create labeling session
session = labeling.create_labeling_session(
    name="labeled_clinical_assistant_review_oct_2025",
    assigned_users=label_users,
    label_schemas=[
        schemas.EXPECTED_FACTS,
        "response_quality"
    ]
)

print(f"Created session: {session.name}")
print(f"Session ID: {session.labeling_session_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC > __Note:__ If you click the below session url, the new Chat-to-review session should be empty for now, but you can use 'Chat' to live chat and vibe test.
# MAGIC
# MAGIC Specifically, 
# MAGIC
# MAGIC Live chat messages in the labeling session (using the "Chat" feature) are not recorded as part of the official human feedback or labeling data. The chat is intended for real-time collaboration and discussion among reviewers, but only explicit labels, comments, and feedback submitted through the labeling UI are stored and associated with the dataset or traces.

# COMMAND ----------

session.url

# COMMAND ----------

# DBTITLE 1,Utility to delete sessions if needed
# import mlflow.genai.labeling as labeling

# # Find the session to delete by name
# all_sessions = labeling.get_labeling_sessions()
# session_to_delete = None
# for session in all_sessions:
#     if session.name == "labeled_clinical_assistant_review_oct_2025":
#         session_to_delete = session
#         break

# if session_to_delete:
#     # Delete the session (removes from Review App)
#     review_app = labeling.delete_labeling_session(session_to_delete)
#     print(f"Deleted session: {session_to_delete.name}")
# else:
#     print("Session not found")

# COMMAND ----------

# MAGIC %md
# MAGIC Since now Chat-to-review session should have no pending need-to-review items, we need to add them by:
# MAGIC
# MAGIC 1. Use add_traces() to add specific, possibly filtered, trace records for review.
# MAGIC 2. Use add_dataset() to add all records from a dataset table for review and labeling.
# MAGIC
# MAGIC You can use both in the same session if you want to review both specific traces and a full dataset

# COMMAND ----------

# DBTITLE 1,Add Traces and Review App URL
# Add traces for labeling
traces = mlflow.search_traces(
    run_id=session.mlflow_run_id
)
session.add_traces(traces)
session.add_dataset(f"{target_catalog_name}.{target_schema_name}.{evaluation_dataset_table_name}")

# Get review app URL
app = labeling.get_review_app()

# We need to explicitly add the agent to use 
if app.agents == None:
    app.add_agent(
        agent_name = "clinical_assistant_agent", 
        model_serving_endpoint= endpoint_name, 
        overwrite=False
    )

print(f"Review app URL: {app.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can proceed with the Chat-to-review session,
# MAGIC 1. __Please open the review app and provide expectation and feedback to at least 1/5 item.__
# MAGIC 2. Then we can sync the evaluation dataset with the human labels (your efforts).

# COMMAND ----------

# DBTITLE 1,Sync the eval_data with your SME labels
from mlflow.genai import datasets
import mlflow

# Sync expectations back to dataset
session.sync(to_dataset=f"{target_catalog_name}.{target_schema_name}.{evaluation_dataset_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we can re-evaluate the agent with your annotated labels

# COMMAND ----------

# DBTITLE 1,re-evaluate with updated dataset
# Use dataset for evaluation
dataset = datasets.get_dataset(f"{target_catalog_name}.{target_schema_name}.{evaluation_dataset_table_name}")

results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=evaluate_model,
    scorers=[
        # With ground truth judges
        Correctness(),
        RetrievalSufficiency(),
        Guidelines(name="clarity", guidelines=guidelines["clarity"]),
        Guidelines(name="accuracy", guidelines=guidelines["accuracy"]),
        # Without ground truth judges
        RelevanceToQuery(),
        Safety(),
        RetrievalGroundedness(),
        RetrievalRelevance(),
        # Custom scorer
        # check_no_pii,
        is_message_relevant
    ]
)

# COMMAND ----------


