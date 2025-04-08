import os
from pathlib import Path
import pandas as pd
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document

import streamlit as st
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()

# ------------------------------
# Initialize LLM and Embeddings
# ------------------------------
llm = AzureOpenAI(
    deployment_name=os.environ["AZURE_COMPLETION_MODEL"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

embed_model = AzureOpenAIEmbedding(
    deployment_name=os.environ["AZURE_EMBEDDING_MODEL"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

Settings.llm = llm
Settings.embed_model = embed_model

# ------------------------------
# Load and index documents
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()

# Load questions and acceptance criteria
question_files = list(DATA_DIR.glob("*.xlsx"))
questions_df = pd.concat([pd.read_excel(f) for f in question_files], ignore_index=True)
# Optional: Print or preview questions_df
print(questions_df.head())


# Combine question and acceptance into one string for better context
reference_texts = [
    f"Q: {row['Question']} | Acceptance: {row['Acceptance Criteria']}"
    for _, row in questions_df.iterrows()
]

# Convert to Documents
reference_docs = [Document(text=txt) for txt in reference_texts]

# Build index
question_reference_index = VectorStoreIndex.from_documents(reference_docs)
question_engine = question_reference_index.as_query_engine(similarity_top_k=1)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Sales Request Assistant", layout="centered")
st.title("🤖 Welcome to Capabilio")
st.markdown("Please describe your idea briefly, and I'll guide you from there. 😊")

# Initialize session message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your request here..."):
    # Show user input
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # RAG-powered answer using documents
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = question_engine.query(prompt)
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})
