import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
from sentence_transformers import SentenceTransformer, util
import requests
import logging
import urllib3
import getpass
import re
from dotenv import load_dotenv
import base64
import openai

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

# Now that environment variables are loaded, you can access them
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CISCO_OPENAI_APP_KEY = os.getenv("CISCO_OPENAI_APP_KEY")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
TOKEN_URL = os.getenv("TOKEN_URL")
CISCO_USER_ID = os.getenv("CISCO_USER_ID", "<YOUR_CISCO_USER_ID>")

# Check for missing environment variables and log them
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
if missing_env_vars:
    log.error(f"Missing required environment variables: {', '.join(missing_env_vars)}")
    # Optionally, you can exit the script if critical variables are missing
    # exit(1)

# Global storage for IDs
stored_data = {
    'device_ids': [],
    'site_ids': [],
}

# RestClientManager class for handling authentication and API requests
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

# Load the Swagger JSON file from a local file
def load_swagger_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Extract relevant API details from Swagger JSON
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

# Generate embeddings for all API descriptions
def generate_api_embeddings(api_endpoints, model):
    api_texts = [f"{api['summary']} {api['description']}" for api in api_endpoints]
    api_embeddings = model.encode(api_texts, convert_to_tensor=True)
    return api_embeddings

# Find the most relevant APIs based on semantic similarity
def find_relevant_api(query, api_endpoints, api_embeddings, model, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, api_embeddings)[0]
    best_matches = torch.topk(cosine_scores, k=top_k)
    matched_apis = [api_endpoints[idx] for idx in best_matches.indices]
    return matched_apis

# Display matched APIs in a readable format with numbering
def display_matched_apis(matched_apis):
    for idx, api in enumerate(matched_apis, start=1):
        print(f"{idx}.")
        print(f"Path: {api['path']}")
        print(f"Method: {api['method']}")
        print(f"Operation ID: {api['operation_id']}")
        print(f"Summary: {api['summary']}")
        print(f"Description: {api['description']}")
        print(f"Tags: {', '.join(api['tags'])}\n")
        print("---")

def display_stored_ids():
    print("\nStored Device IDs:")
    for idx, dev_id in enumerate(stored_data['device_ids'], start=1):
        print(f"{idx}. {dev_id}")
    print("Stored Site IDs:")
    for idx, site_id in enumerate(stored_data['site_ids'], start=1):
        print(f"{idx}. {site_id}")
    print()

def get_value_from_selection(selection, stored_list, placeholder):
    if selection.isdigit():
        selection = int(selection)
        if 1 <= selection <= len(stored_list):
            return stored_list[selection - 1]
        else:
            print("Invalid selection.")
            return input(f"Please enter value for '{placeholder}': ").strip()
    else:
        return selection  # User entered a custom ID

# Execute the API call and display results
def execute_api(api, rest_client):
    print(f"Executing API: {api['operation_id']}")
    try:
        # Check for placeholders in the path
        path = api['path']
        placeholders = re.findall(r'\{(.*?)\}', path)
        if placeholders:
            for placeholder in placeholders:
                placeholder_lower = placeholder.lower()
                # Ask user if they want to see stored IDs
                show_ids_decision = input(f"Do you want to see stored IDs for '{placeholder}'? (yes/no): ").strip().lower()
                if show_ids_decision == 'yes':
                    display_stored_ids()
                value = input(f"Please enter value for '{placeholder}': ").strip()
                # Replace the placeholder with the value in the path
                path = path.replace(f'{{{placeholder}}}', value)

        if api['method'].lower() == 'get':
            response = rest_client.get_api(path)
        elif api['method'].lower() == 'post':
            # For POST requests, you might need to supply payload data
            payload = {}
            need_payload = input("This API requires a payload. Would you like to provide it? (yes/no): ").strip().lower()
            if need_payload == 'yes':
                # Prompt user for payload details
                print("Enter payload data as key-value pairs. Type 'done' when finished.")
                while True:
                    key = input("Enter payload field name (or 'done' to finish): ").strip()
                    if key.lower() == 'done':
                        break
                    if key.lower() == 'show ids':
                        display_stored_ids()
                        continue
                    value = input(f"Enter value for '{key}': ").strip()
                    if value.lower() == 'show ids':
                        display_stored_ids()
                        value = input(f"Enter value for '{key}': ").strip()
                    payload[key] = value
            else:
                print("Using empty payload.")
            response = rest_client.post_api(path, payload)
        else:
            print(f"HTTP method {api['method']} not supported for execution.")
            return

        # Pretty print the response
        print("\nAPI Response:")
        print(json.dumps(response, indent=4))

        # Extract and store IDs
        extract_and_store_ids(response)

        # Ask if the user wants a summary
        summarize_decision = input("Would you like GPT-4 to summarize the API response? (yes/no): ").strip().lower()
        if summarize_decision == 'yes':
            summary = summarize_response(response)
            print("\nGPT-4 Summary of the API Response:")
            print(summary)
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        print(f"Response content: {err.response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_and_store_ids(response):
    if isinstance(response, dict) and 'response' in response:
        items = response['response']
        if isinstance(items, list):
            for item in items:
                store_ids_from_item(item)
        elif isinstance(items, dict):
            store_ids_from_item(items)

def store_ids_from_item(item):
    # Extract device ID
    device_id = item.get('id')
    if device_id and device_id not in stored_data['device_ids']:
        stored_data['device_ids'].append(device_id)
        print(f"Stored device ID: {device_id}")
    # Extract site ID
    site_id = item.get('siteId')
    if site_id and site_id not in stored_data['site_ids']:
        stored_data['site_ids'].append(site_id)
        print(f"Stored site ID: {site_id}")

# Summarize the API response using Azure OpenAI
def summarize_response(response):
    # Convert response to a string if it's not already
    if not isinstance(response, str):
        response_text = json.dumps(response, indent=2)
    else:
        response_text = response

    # Prepare the prompt for GPT-4
    prompt = f"Summarize the following API response in clear, concise language:\n\n{response_text}"

    # Load environment variables (already loaded at the beginning)

    # Check for missing variables
    required_vars = {
        "CLIENT_ID": CLIENT_ID,
        "CLIENT_SECRET": CLIENT_SECRET,
        "CISCO_OPENAI_APP_KEY": CISCO_OPENAI_APP_KEY,
        "DEPLOYMENT_NAME": DEPLOYMENT_NAME,
        "AZURE_ENDPOINT": AZURE_ENDPOINT,
        "API_VERSION": API_VERSION,
        "TOKEN_URL": TOKEN_URL,
        "CISCO_USER_ID": CISCO_USER_ID,
    }

    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        log.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return "No summary available."

    # Obtain access token
    try:
        # Prepare the token request
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

    # Set OpenAI API configuration
    try:
        openai.api_type = "azure"
        openai.api_key = access_token  # Use the obtained token
        openai.api_base = AZURE_ENDPOINT
        openai.api_version = API_VERSION

        # Prepare the user header
        user_header = {
            "appkey": CISCO_OPENAI_APP_KEY,
            "user": CISCO_USER_ID
        }
        user_header_json = json.dumps(user_header)

        # Make the OpenAI API call using ChatCompletion
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

# Hybrid matching function to check expected questions first, then fallback to full search
def hybrid_search(
    query, expected_questions, question_embeddings,
    api_endpoints, api_embeddings, model, rest_client,
    similarity_threshold=0.7, top_k=5
):
    # Normalize the query
    query = query.lower().strip()

    # Generate embedding for the user query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Check similarity with expected questions
    question_similarity_scores = util.cos_sim(query_embedding, question_embeddings)[0]
    best_match_idx = torch.argmax(question_similarity_scores)
    best_score = question_similarity_scores[best_match_idx].item()
    print(f"Best similarity score with expected questions: {best_score}")

    # Check if similarity is above the threshold
    if best_score > similarity_threshold:
        best_question = list(expected_questions.keys())[best_match_idx]
        matched_api = expected_questions[best_question]
        print(f"\nMatched API directly from expected question: '{best_question}'")
        print(f"Path: {matched_api['path']}")
        print(f"Method: {matched_api['method']}")
        print(f"Operation ID: {matched_api['operation_id']}")
        print(f"Summary: {matched_api['summary']}")
        print(f"Description: {matched_api['description']}")
        print(f"Tags: {', '.join(matched_api['tags'])}\n")
        print("---")
        # Ask the user if they want to execute the matched API
        execute_decision = input("Would you like to execute this API? (yes/no): ").strip().lower()
        if execute_decision == 'yes':
            execute_api(matched_api, rest_client)
        else:
            # Ask if the user wants to perform a wider search
            search_decision = input("Would you like to perform a wider search across all APIs? (yes/no): ").strip().lower()
            if search_decision == 'yes':
                # Perform a full API search
                matched_apis = find_relevant_api(query, api_endpoints, api_embeddings, model, top_k=top_k)
                display_matched_apis(matched_apis)
                # Prompt the user to select an API to execute
                if matched_apis:
                    print(f"\nEnter the number of the API you wish to execute (1-{len(matched_apis)}), or 0 to skip:")
                    while True:
                        selection = input("Your choice: ").strip()
                        if selection.isdigit():
                            selection = int(selection)
                            if 0 <= selection <= len(matched_apis):
                                break
                        print(f"Please enter a number between 0 and {len(matched_apis)}.")
                    if selection == 0:
                        print("Skipping API execution.")
                    else:
                        selected_api = matched_apis[selection - 1]
                        execute_api(selected_api, rest_client)
                else:
                    print("No APIs found to execute.")
            else:
                print("Okay, no action will be taken.")
    else:
        # If no close match, fallback to full API search
        print("\nNo close match found in expected questions. Performing full API search...")
        matched_apis = find_relevant_api(query, api_endpoints, api_embeddings, model, top_k=top_k)
        display_matched_apis(matched_apis)
        # Prompt the user to select an API to execute
        if matched_apis:
            print(f"\nEnter the number of the API you wish to execute (1-{len(matched_apis)}), or 0 to skip:")
            while True:
                selection = input("Your choice: ").strip()
                if selection.isdigit():
                    selection = int(selection)
                    if 0 <= selection <= len(matched_apis):
                        break
                print(f"Please enter a number between 0 and {len(matched_apis)}.")
            if selection == 0:
                print("Skipping API execution.")
            else:
                selected_api = matched_apis[selection - 1]
                execute_api(selected_api, rest_client)
        else:
            print("No APIs found to execute.")

# Main function to set up and continuously search APIs
def main():
    # Server credentials (used for authentication and API calls)
    server = input("Enter the server address (e.g., 10.85.116.58): ").strip()
    authenticated = False
    while not authenticated:
        username = input("Enter your username: ").strip()
        password = getpass.getpass("Enter your password: ")
        try:
            # Initialize RestClientManager
            rest_client = RestClientManager(server, username, password)
            authenticated = True  # Set flag to True if authentication succeeds
        except requests.exceptions.HTTPError as err:
            print(f"Authentication failed: {err}")
            retry = input("Would you like to try again? (yes/no): ").strip().lower()
            if retry != 'yes':
                print("Exiting...")
                return  # Exit the program
        except Exception as e:
            print(f"An error occurred: {e}")
            return  # Exit the program

    # Load Swagger JSON from a local file
    swagger_filepath = "swagger.json"  # Provide the correct path to your local Swagger JSON file
    swagger_data = load_swagger_json(swagger_filepath)

    # Extract and embed APIs
    api_endpoints = extract_api_endpoints(swagger_data)
    model = SentenceTransformer('all-mpnet-base-v2')  # Load a powerful model for embedding
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
        },
        # Add more expected question-to-API mappings as needed
    }

    # Precompute embeddings for expected questions
    question_texts = [q.lower().strip() for q in expected_questions.keys()]
    question_embeddings = model.encode(question_texts, convert_to_tensor=True)

    # Continuous query loop
    print("\nWelcome to the Cisco DNAC API Chatbot! Type 'exit' to quit.\n")
    while True:
        user_query = input("Enter your question about Cisco DNAC APIs (or type 'list ids' to view stored IDs): ")

        # Check if the user wants to exit or list IDs
        if user_query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        elif user_query.lower() == 'list ids':
            display_stored_ids()
            continue

        # Run hybrid search
        hybrid_search(
            user_query, expected_questions, question_embeddings,
            api_endpoints, api_embeddings, model, rest_client,
            similarity_threshold=0.7
        )

        print("\n---\n")

if __name__ == "__main__":
    main()
