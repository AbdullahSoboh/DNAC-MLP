import os
from dotenv import load_dotenv
import openai
import base64
import requests
from langchain_openai import AzureChatOpenAI
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

class LLM:
    def __init__(self):
        self.prompt = None
        self.token = None
        openai.api_type = "azure"
        openai.api_version = os.getenv("API_VERSION")
        openai.api_base = os.getenv("AZURE_ENDPOINT")
        self.cisco_idp = os.getenv("TOKEN_URL")
        self.client_id = os.getenv("CLIENT_ID")
        self.app_key = os.getenv("CISCO_OPENAI_APP_KEY")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.model = os.getenv("DEPLOYMENT_NAME")
        self.llm = None
        self.cisco_user_id = "<YOUR_CISCO_ID>"
        self.parser = StrOutputParser()

        if self.token is not None:
            openai.api_key = self.token


    def get_auth_token(self):

        payload = "grant_type=client_credentials"
        value = base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode('utf-8')).decode('utf-8')
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}"
        }

        token_response = requests.request("POST", self.cisco_idp, headers=headers, data=payload)

        self.token = token_response.json()["access_token"]
    
        openai.api_key = self.token

    
    def get_cisco_user_id(self):
        return self.cisco_user_id

    def configure_llm_chain(self):

        self.get_auth_token()

        self.llm = AzureChatOpenAI(deployment_name=self.model, 
                            azure_endpoint = openai.api_base, 
                            api_key=openai.api_key,  
                            api_version=openai.api_version,
                            model_kwargs=dict(
                            user=f'{{"appkey": "{self.app_key}", "user": "{self.cisco_user_id}"}}'
                            )
            )
        return self.llm | self.parser



st.title("Sample App")

llm = LLM()
llm_chain = llm.configure_llm_chain()


if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
    ]

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)



text = st.chat_input(
    "I am an Helpful Assitant. Ask me anything...",
)

st.session_state.llm = llm_chain

if text:
    st.session_state.messages.append(HumanMessage(content=text))
    st.chat_message("human").write(text)
    with st.chat_message("assistant"):
        ai_answer = llm_chain.invoke(st.session_state.messages)
        st.write(ai_answer)
        st.session_state.messages.append(AIMessage(content=ai_answer))
    