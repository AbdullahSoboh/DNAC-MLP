import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
import requests
import logging
import urllib3
import re
from dotenv import load_dotenv
import base64
import openai
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

# Environment variables
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CISCO_OPENAI_APP_KEY = os.getenv("CISCO_OPENAI_APP_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
TOKEN_URL = os.getenv("TOKEN_URL")
CISCO_USER_ID = os.getenv("CISCO_USER_ID", "<YOUR_CISCO_USER_ID>")

required_env_vars = {
    "CLIENT_ID": CLIENT_ID,
    "CLIENT_SECRET": CLIENT_SECRET,
    "CISCO_OPENAI_APP_KEY": CISCO_OPENAI_APP_KEY,
    "DEPLOYMENT_NAME": DEPLOYMENT_NAME,
    "AZURE_ENDPOINT": AZURE_ENDPOINT,
    "API_VERSION": API_VERSION,
    "TOKEN_URL": TOKEN_URL,
    "CISCO_USER_ID": CISCO_USER_ID,
}

missing_env_vars = [var for var, value in required_env_vars.items() if not value]
# if missing_env_vars:
#     st.error(f"Missing required environment variables: {', '.join(missing_env_vars)}")

# Global storage for IDs
stored_data = {
    'device_ids': [],
    'site_ids': [],
}

class RestClientManager:
    def __init__(self, server, username, password):
        self.base_url = f"https://{server}"
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification if necessary
        self.token = self.get_token(username, password)
        self.session.headers.update({'X-Auth-Token': self.token})

    def get_token(self, username, password):
        url = f"{self.base_url}/dna/system/api/v1/auth/token"
        response = self.session.post(url, auth=(username, password))
        response.raise_for_status()
        token = response.json()['Token']
        log.info("Authentication successful. Token obtained.")
        return token

    def get_api(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def post_api(self, endpoint, payload):
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()

def load_swagger_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def extract_api_endpoints(swagger_data):
    api_endpoints = []
    for path, methods in swagger_data.get("paths", {}).items():
        for method, details in methods.items():
            api_endpoints.append({
                "path": path,
                "method": method,
                "operation_id": details.get("operationId", ""),
                "summary": details.get("summary", ""),
                "description": details.get("description", ""),
                "tags": details.get("tags", [])
            })
    return api_endpoints

def generate_api_embeddings(api_endpoints, model):
    api_texts = [f"{api['summary']} {api['description']}" for api in api_endpoints]
    api_embeddings = model.encode(api_texts, convert_to_tensor=True)
    return api_embeddings

def find_relevant_api(query, api_endpoints, api_embeddings, model, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, api_embeddings)[0]
    best_matches = torch.topk(cosine_scores, k=top_k)
    matched_apis = [api_endpoints[idx] for idx in best_matches.indices]
    return matched_apis

def display_matched_apis(matched_apis):
    for idx, api in enumerate(matched_apis, start=1):
        st.write(f"{idx}.")
        st.write(f"**Path**: {api['path']}")
        st.write(f"**Method**: {api['method']}")
        st.write(f"**Operation ID**: {api['operation_id']}")
        st.write(f"**Summary**: {api['summary']}")
        st.write(f"**Description**: {api['description']}")
        st.write(f"**Tags**: {', '.join(api['tags'])}\n")
        st.write("---")

def display_stored_ids():
    st.write("Stored Device IDs:")
    for idx, dev_id in enumerate(stored_data['device_ids'], start=1):
        st.write(f"{idx}. {dev_id}")
    st.write("Stored Site IDs:")
    for idx, site_id in enumerate(stored_data['site_ids'], start=1):
        st.write(f"{idx}. {site_id}")

def execute_api(api, rest_client):
    st.write(f"Executing API: {api['operation_id']}")
    try:
        path = api['path']
        placeholders = re.findall(r'\{(.*?)\}', path)
        if placeholders:
            for placeholder in placeholders:
                if st.radio(f"Do you want to see stored IDs for '{placeholder}'?", ('yes', 'no')) == 'yes':
                    display_stored_ids()
                value = st.text_input(f"Enter value for '{placeholder}': ")
                path = path.replace(f'{{{placeholder}}}', value)

        if api['method'].lower() == 'get':
            response = rest_client.get_api(path)
        elif api['method'].lower() == 'post':
            payload = {}
            if st.radio("This API requires a payload. Would you like to provide it?", ('yes', 'no')) == 'yes':
                st.write("Enter payload data as key-value pairs.")
                while True:
                    key = st.text_input("Enter payload field name (or 'done' to finish): ")
                    if key.lower() == 'done':
                        break
                    value = st.text_input(f"Enter value for '{key}': ")
                    payload[key] = value
            else:
                st.write("Using empty payload.")
            response = rest_client.post_api(path, payload)
        else:
            st.write(f"HTTP method {api['method']} not supported for execution.")
            return

        st.write("API Response:")
        st.json(response)

        extract_and_store_ids(response)

        if st.radio("Would you like GPT-4 to summarize the API response?", ('yes', 'no')) == 'yes':
            summary = summarize_response(response)
            st.write("GPT-4 Summary of the API Response:")
            st.write(summary)
    except requests.exceptions.HTTPError as err:
        st.write(f"HTTP error occurred: {err}")
        st.write(f"Response content: {err.response.text}")
    except Exception as e:
        st.write(f"An error occurred: {e}")

def extract_and_store_ids(response):
    if isinstance(response, dict) and 'response' in response:
        items = response['response']
        if isinstance(items, list):
            for item in items:
                store_ids_from_item(item)
        elif isinstance(items, dict):
            store_ids_from_item(items)

def store_ids_from_item(item):
    device_id = item.get('id')
    if device_id and device_id not in stored_data['device_ids']:
        stored_data['device_ids'].append(device_id)
        st.write(f"Stored device ID: {device_id}")
    site_id = item.get('siteId')
    if site_id and site_id not in stored_data['site_ids']:
        stored_data['site_ids'].append(site_id)
        st.write(f"Stored site ID: {site_id}")

def summarize_response(response):
    response_text = json.dumps(response, indent=2) if not isinstance(response, str) else response

    prompt = f"Summarize the following API response:\n\n{response_text}"

    missing_vars = [var for var, value in required_env_vars.items() if not value]
    if missing_vars:
        log.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return "No summary available."

    try:
        payload = "grant_type=client_credentials"
        value = base64.b64encode(f'{CLIENT_ID}:{CLIENT_SECRET}'.encode('utf-8')).decode('utf-8')
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}"
        }
        token_response = requests.post(TOKEN_URL, headers=headers, data=payload)
        token_response.raise_for_status()
        access_token = token_response.json()['access_token']
        log.info("Access token obtained successfully.")
    except Exception as e:
        log.error(f"Error obtaining access token: {e}")
        return "No summary available."

    try:
        openai.api_type = "azure"
        openai.api_key = access_token
        openai.api_base = AZURE_ENDPOINT
        openai.api_version = API_VERSION

        user_header = {
            "appkey": CISCO_OPENAI_APP_KEY,
            "user": CISCO_USER_ID
        }
        user_header_json = json.dumps(user_header)

        response = openai.ChatCompletion.create(
            engine=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Well versed in DNAC Cisco APIs."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5,
            user=user_header_json
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        log.error(f"An error occurred while summarizing: {e}")
        return "No summary available."

def hybrid_search(
    query, expected_questions, question_embeddings,
    api_endpoints, api_embeddings, model, rest_client,
    similarity_threshold=0.7, top_k=5
):
    query = query.lower().strip()
    st.write(f"Query - {query}")
    query_embedding = model.encode(query, convert_to_tensor=True)
    question_similarity_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_match_idx = torch.argmax(question_similarity_scores)
    best_score = question_similarity_scores[best_match_idx].item()
    st.write(f"Best similarity score with expected questions: {best_score}")

    if best_score > similarity_threshold:
        best_question = list(expected_questions.keys())[best_match_idx]
        matched_api = expected_questions[best_question]
        st.write(f"\nMatched API directly from expected question: '{best_question}'")
        st.write(f"Path: {matched_api['path']}")
        st.write(f"Method: {matched_api['method']}")
        st.write(f"Operation ID: {matched_api['operation_id']}")
        st.write(f"Summary: {matched_api['summary']}")
        st.write(f"Description: {matched_api['description']}")
        st.write(f"Tags: {', '.join(matched_api['tags'])}\n")
        st.write("---")
        if st.radio("Would you like to execute this API?", ('yes', 'no')) == 'yes':
            execute_api(matched_api, rest_client)
        else:
            if st.radio("Would you like to perform a wider search across all APIs?", ('yes', 'no')) == 'yes':
                matched_apis = find_relevant_api(query, api_endpoints, api_embeddings, model, top_k=top_k)
                display_matched_apis(matched_apis)
                if matched_apis:
                    selection = st.number_input(f"Enter the number of the API you wish to execute (1-{len(matched_apis)}), or 0 to skip:", min_value=0, max_value=len(matched_apis))
                    if selection > 0:
                        selected_api = matched_apis[selection - 1]
                        execute_api(selected_api, rest_client)
                    else:
                        st.write("Skipping API execution.")
                else:
                    st.write("No APIs found to execute.")
            else:
                st.write("Okay, no action will be taken.")
    else:
        st.write("\nNo close match found in expected questions. Performing full API search...")
        matched_apis = find_relevant_api(query, api_endpoints, api_embeddings, model, top_k=top_k)
        display_matched_apis(matched_apis)
        if matched_apis:
            selection = st.number_input(f"Enter the number of the API you wish to execute (1-{len(matched_apis)}), or 0 to skip:", min_value=0, max_value=len(matched_apis))
            if selection > 0:
                selected_api = matched_apis[selection - 1]
                execute_api(selected_api, rest_client)
            else:
                st.write("Skipping API execution.")
        else:
            st.write("No APIs found to execute.")

def main():
    server = st.text_input("Enter the server address (e.g., 10.85.116.58):").strip()
    username = st.text_input("Enter your username:").strip()
    password = st.text_input("Enter your password:", type="password").strip()

    authenticated = False
    if st.button("Authenticate"):
        try:
            rest_client = RestClientManager(server, username, password)
            authenticated = True
            st.success("Authentication successful.")
        except requests.exceptions.HTTPError as err:
            st.error(f"Authentication failed: {err}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if authenticated:
        swagger_filepath = "swagger.json"
        swagger_data = load_swagger_json(swagger_filepath)

        api_endpoints = extract_api_endpoints(swagger_data)
        model = SentenceTransformer('all-mpnet-base-v2')
        api_embeddings = generate_api_embeddings(api_endpoints, model)

        # Expected questions and their mappings to specific APIs
        expected_questions = {
            "provision a device to a site": {
                "path": "/dna/intent/api/v1/sda/provisionDevices",
                "method": "post",
                "operation_id": "provisionDevices",
                "summary": "Provision devices",
                "description": "Provisions network devices to respective Sites based on user input.",
                "tags": ["SDA"]
            },
            "get devices assigned to a site": {
                "path": "/dna/intent/api/v1/site-member/{id}/member",
                "method": "get",
                "operation_id": "getDevicesThatAreAssignedToASite",
                "summary": "Get devices that are assigned to a site",
                "description": "API to get devices that are assigned to a site.",
                "tags": ["Sites"]
            },
            "what are devices": {
                "path": "/dna/intent/api/v1/sda/provisionDevices",
                "method": "get",
                "operation_id": "getProvisionedDevices",
                "summary": "Get provisioned devices",
                "description": "Returns the list of provisioned devices based on query parameters.",
                "tags": ["SDA"]
            },
            "Build a site network from json": {
                "path": "/dna/intent/api/v1/sda/provisionDevices",
                "method": "get",
                "operation_id": "getProvisionedDevices",
                "summary": "Get provisioned devices",
                "description": "Returns the list of provisioned devices based on query parameters.",
                "tags": ["SDA"]
            }
        }

        question_texts = [q.lower().strip() for q in expected_questions.keys()]
        question_embeddings = model.encode(question_texts, convert_to_tensor=True)

        user_query = st.text_input("Enter your question about Cisco DNAC APIs:")
        if user_query:
            hybrid_search(
                user_query, expected_questions, question_embeddings,
                api_endpoints, api_embeddings, model, rest_client,
                similarity_threshold=0.7
            )

if __name__ == "__main__":
    st.title("Cisco DNAC API Agent")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
        ]
    main()