import streamlit as st
import os
import re
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from pathlib import Path
from llama_index.core.llms import ChatMessage

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI LLM
llm = AzureOpenAI(
    deployment_name=os.environ["AZURE_COMPLETION_MODEL"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

def load_base_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent.parent / "Data" / "Q_A_Natural.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()

def extract_first_follow_up_question(response_text: str) -> str:
    sentences = re.split(r'(?<=[.?!])\s+', response_text.strip())
    for s in sentences:
        if s.endswith("?"):
            return s
    return ""

criteria_checklist = load_base_prompt()

st.set_page_config(page_title="Sales Request Assistant", layout="centered")
st.title("🤖 Welcome to Capabilio")
st.markdown("Please describe your idea briefly, and I'll guide you from there. 😊")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_request" not in st.session_state:
    st.session_state.user_request = ""

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your request here..."):
    # Show and save user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.user_request += prompt + "\n"

    # Build prompt
    full_prompt = criteria_checklist.replace(
        'User Request:\n"""', f'User Request:\n"""\n{st.session_state.user_request}\n'
    )

    # Query LLM
    with st.spinner("Analyzing request against business criteria..."):
        response = llm.complete(full_prompt)

    # Show assistant response
    st.chat_message("assistant").markdown(response.text)
    st.session_state.messages.append({"role": "assistant", "content": response.text})

    # If success, stop. Else, extract first follow-up and re-prompt
    if "Success!" in response.text:
        st.success("✅ All business criteria fulfilled. Great job!")
    else:
        follow_up = extract_first_follow_up_question(response.text)
        if follow_up:
            st.chat_message("assistant").markdown(follow_up)
            st.session_state.messages.append({"role": "assistant", "content": follow_up})
