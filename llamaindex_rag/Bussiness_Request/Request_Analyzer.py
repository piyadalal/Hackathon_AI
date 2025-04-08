
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
from llama_index.core.llms import ChatMessage
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
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant that collects sales request information by asking smart questions and guiding the user through 18 required fields."}
    ]

# Display previous messages
for msg in st.session_state.messages[1:]:  # skip system
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Please describe your request"):
    # Show user input
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Prepare messages in llama_index format
    history = [
        ChatMessage(role=m["role"], content=m["content"]) for m in st.session_state.messages
    ]

    # Call AzureOpenAI LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.chat(messages=history)
            st.markdown(response.message.content)

    # Save response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.message.content})
