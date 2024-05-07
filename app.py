
#%%
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.voyageai import VoyageEmbedding
from llama_index.llms.anthropic import Anthropic
from qdrant_client import QdrantClient
import json

# Load secrets
with open("../secrets.json", "r") as f:
    secrets = json.load(f)
API_KEYS = secrets["keys"]

#%%

# Initialize settings
model_name = "voyage-large-2"
voyage_api_key = API_KEYS["voyage-ai"]
embed_model = VoyageEmbedding(model_name=model_name, voyage_api_key=voyage_api_key)
Settings.embed_model = embed_model

tokenizer = Anthropic().tokenizer
model_name = secrets["config"]["anthropic"]["model-sonnet"]
anthropic_api_key = API_KEYS["anthropic"]
llm = Anthropic(model=model_name, api_key=anthropic_api_key)
Settings.tokenizer = tokenizer
Settings.llm = llm
Settings.chunk_size = 1024

client = QdrantClient(
    host=secrets["config"]["qdrant-cloud"]["host"],
    port=secrets["config"]["qdrant-cloud"]["port"],
    api_key=API_KEYS["qdrant-cloud"],
)

#%%
# Load collections
with open(r"D:\Genesilico_github\OncologistCoPilot\chat\collections.json", "r") as f:
    data_loader_config = json.load(f)

query_engine_tools = []
for cfg in data_loader_config:
    collection_name = cfg['collection_name']
    description = cfg["description"]
    vector_store = QdrantVectorStore(collection_name, client=client, enable_hybrid=True, batch_size=20)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(similarity_top_k=10, sparse_top_k=15, response_mode="tree_summarize")
    query_engine_tools.append(
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(name=collection_name, description=description)
        )
    )

# Initialize ReActAgent
agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True, max_iterations=20)

#%%

# Define Streamlit UI
st.title("Breast Cancer Treatment Advisor")

history = st.text_area("Brief History", "")

if st.button("Get Treatment Recommendation"):
    response = agent.chat(history)
    st.write("### Treatment Recommendation")
    st.write(response)

# %%
