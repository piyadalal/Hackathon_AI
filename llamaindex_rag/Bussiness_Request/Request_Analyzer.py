import os
import tempfile
from pathlib import Path
import openai
import os

import streamlit as st
import numpy as np
import weaviate
from dotenv import load_dotenv
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI LLM
llm = AzureOpenAI(
    deployment_name=os.environ["AZURE_COMPLETION_MODEL"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

# Initialize Azure OpenAI Embeddings
embed_model = AzureOpenAIEmbedding(
    deployment_name=os.environ["AZURE_EMBEDDING_MODEL"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

# Assign global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Connect to local Weaviate instance
client = weaviate.connect_to_local()
collection_name = "RequestAnalyzer"

# Clear existing collection for a fresh start
if client.collections.exists(collection_name):
    client.collections.delete(collection_name)

st.set_page_config(page_title="Sales Request Assistant", layout="centered")
st.title("🤖 Sales Request Assistant")
st.markdown("Please provide details about your request. I will guide you through a few questions.")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

