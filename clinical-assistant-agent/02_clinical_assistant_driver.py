# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export modified with additional features for evaluation. 
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://docs.databricks.com/generative-ai/agent-framework/build-genai-apps.html) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - All `TODO`s in this notebook you can skip. We already make them work for this example.
# MAGIC - They are for later expansion, e.g., different tools.

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html).

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

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_node import ToolNode
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC
# MAGIC LLM_ENDPOINT_NAME = "databricks-claude-3-7-sonnet"
# MAGIC # LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC system_prompt = """You are a healthcare assistant specialized in analyzing real-world healthcare data. You will be asked questions about patient enrollment, medical claims, pharmacy claims, diagnoses, and procedures from HealthVerity's healthcare dataset. You should only answer questions relevant to this topic and should politely decline to answer any off topic questions. Be concise and clear - no need to repeat the question.
# MAGIC
# MAGIC Use the tools at your disposal to answer the user's question. If you don't know the answer, say so. If the tools fail to execute, say so, and say why if you can. If it isn't clear which tool should be used, ask the user and summarize the tools that you can use.
# MAGIC
# MAGIC Available tools include:
# MAGIC - get_patient_enrollment: Get patient demographics and enrollment information
# MAGIC - get_medical_claims: Get medical claims for a patient on a specific service date
# MAGIC - get_patient_diagnoses: Get all diagnosis codes for a patient
# MAGIC - get_pharmacy_claims: Get pharmacy claims and medication history for a patient
# MAGIC - get_patient_procedures: Get procedure codes and details for a patient
# MAGIC
# MAGIC If there is a request including a DATE, please always return the full date when possible. 
# MAGIC """
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC #You can use UDFs in Unity Catalog as agent tools
# MAGIC # HealthVerity clinical assistant tools
# MAGIC uc_tool_names = [f"{target_catalog_name}.{target_schema_name}.*"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC # # (Optional) Use Databricks vector search indexes as tools
# MAGIC # # See https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html
# MAGIC # # for details
# MAGIC #
# MAGIC # # TODO: Add vector search indexes as tools or delete this block
# MAGIC # vector_search_tools = [
# MAGIC #         VectorSearchRetrieverTool(
# MAGIC #         index_name="",
# MAGIC #         # filters="..."
# MAGIC #     )
# MAGIC # ]
# MAGIC # tools.extend(vector_search_tools)
# MAGIC
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[Sequence[BaseTool], ToolNode],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     @mlflow.trace(name="agent_call_model")
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     # Create a custom tool node with tracing
# MAGIC     @mlflow.trace(name="agent_tool_execution")
# MAGIC     def call_tools(state: ChatAgentState, config: RunnableConfig):
# MAGIC         tool_node = ChatAgentToolNode(tools)
# MAGIC         return tool_node.invoke(state, config)
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", RunnableLambda(call_tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     @mlflow.trace(name="agent_predict")
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     @mlflow.trace(name="agent_predict_stream")
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

added_var_content = f"""target_catalog_name = '{target_catalog_name}'\n
target_schema_name = '{target_schema_name}'\n
"""
with open("agent.py", "r") as f:
    existing_content = f.read()

with open("agent.py", "w") as f:
    f.write(added_var_content + existing_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

dbutils.library.restartPython()

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

# COMMAND ----------

# MAGIC %md
# MAGIC Note:
# MAGIC **Mlflow experiement needs to be registered into your workspace folder (use absolute path like below). Repo folder wont work.**

# COMMAND ----------

# DBTITLE 1,make sure to provide a local absolute path
import mlflow

#: plan a, setup with default artifact location on managed mlflow-tracking server
experiment_info = mlflow.set_experiment(experiment_path)

#: plan b, setup with your DIY, use mlflow.create_experiment()
# Create with custom configuration
# experiment_path = mlflow.create_experiment(
#     "production-models",
#     artifact_location="s3://my-bucket/experiments/",
#     tags={"team": "data-science", "environment": "prod"},
# )

# COMMAND ----------

experiment_info

# COMMAND ----------

from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": f"What is the healthcare journey for patient {patient_id}?"}]})

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": f"What is the latest medication has patient {patient_id} been prescribed? What is its diagnosis code?"}]}
):
    print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog Function queries a [vector search index](https://docs.databricks.com/generative-ai/agent-framework/unstructured-retrieval-tools.html) or leverages [external functions](https://docs.databricks.com/generative-ai/agent-framework/external-connection-tools.html), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://docs.databricks.com/generative-ai/agent-framework/log-agent.html#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

from agent import tools, LLM_ENDPOINT_NAME

# COMMAND ----------

tools

# COMMAND ----------

LLM_ENDPOINT_NAME

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        # TODO: If the UC function includes dependencies like external connection or vector search, please include them manually.
        # See the TODO in the markdown above for more information.
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": f"What is the healthcare journey for patient {patient_id}?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        input_example=input_example,
        resources=resources,
        pip_requirements="requirements.txt"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://docs.databricks.com/machine-learning/model-serving/model-serving-debug.html#validate-inputs) for details

# COMMAND ----------

# DBTITLE 1,env_manager please use 'uv'
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": f"What enrollment information do you have for patient {patient_id}?"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

# MAGIC %pip install mlflow[databricks]

# COMMAND ----------

# DBTITLE 1,make sure you pip installed package mlflow[databricks]
mlflow.set_registry_uri("databricks-uc")

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=model_uc_name
)

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

client.set_registered_model_alias(model_uc_name, "Champion", uc_registered_model_info.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

# DBTITLE 1,Uncomment to create endpoint if needed
from databricks import agents

agents.deploy(
    model_uc_name,
    uc_registered_model_info.version,
    tags={"RemoveAfter": "10-31-2025"},
    scale_to_zero=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See [docs](https://docs.databricks.com/generative-ai/deploy-agent.html) for details

# COMMAND ----------

# MAGIC %md
# MAGIC ## We can also work with a Genie room with optimized Text-2-SQL on UC tables augmented with AI generated table metadata /comments!

# COMMAND ----------

from IPython.display import HTML, display

# Create HTML link to AI Playground
html_link = f'<a href="https://e2-demo-field-eng.cloud.databricks.com/genie/rooms/01f0574f6d4f11558af894a950b8bf19" target="_blank">Go to Genie Room</a>'
display(HTML(html_link))

# COMMAND ----------

# MAGIC %md
# MAGIC ## We can also do Key Information Extraction using Agent Bricks on all of our clinical notes. 

# COMMAND ----------

# Create HTML link to AI Playground
html_link = f'<a href="https://e2-demo-field-eng.cloud.databricks.com/ml/bricks/kie/use/9a577aad-4adc-4a33-9dbb-88883e6c66b8?o=1444828305810485">Go to Agent Bricks</a>'
display(HTML(html_link))

# COMMAND ----------


