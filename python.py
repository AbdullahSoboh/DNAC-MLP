import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
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
import time
from requests.auth import HTTPBasicAuth

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

# Global storage for IDs and other data
stored_data = {
    'device_ids': [],
    'site_ids': [],
    'cli_id': None,
    'snmp_read_id': None,
    'snmp_write_id': None,
}

# RestClientManager class
class RestClientManager:
    """ Client manager to interact with Rest API Client. """

    # Increasing timeout to 60 sec as we are seeing read timeouts
    TIMEOUT = 60  # As requested by Maglev team via Olaf

    def __init__(self,
                 server,
                 username,
                 password,
                 version="v1",
                 connect=True,
                 force=False,
                 port=None
                 ):
        """ Initializer for Rest
        Args:
            server (str)   : cluster server name (routable DNS address or IP)
            username (str) : user name to authenticate with
            password (str) : password to authenticate with
            port (str)     : port number
            connect (bool) : Connects if True
            force (bool)   : Connects forcefully if it is set to True
        """

        base_url = ""
        protocol = "https"
        self.version = version
        self.server = server
        self.username = username
        self.password = password
        self.port = str(port) if port else ""

        if port:
            self.server = self.server + ":" + self.port

        if protocol not in ["http", "https"]:
            log.error("Not supported protocol {}.".format(protocol))
            raise Exception("Not supported protocol {}.".format(protocol))

        self.base_url = "{}://{}{}".format(protocol, self.server, base_url)
        self.endpoint_base_url = base_url

        self._default_headers = {}
        self._common_headers = {}
        self.compact_response = {}

        self.__connected = False
        if connect:
            self.connect(force=force)

    def get_api(self, resource_path, payload=None, timeout=TIMEOUT, port=None, protocol=None, **kwargs):
        if self.__connected:
            print("\nChecking the Token Validity\n")
            self.__check_for_token_refresh()

        headers = {"Content-Type": "application/json", **self.common_headers}
        request_url = self.base_url + resource_path

        # log.info(f"\nGet API, Request URL: {request_url}")

        response = requests.get(request_url, headers=headers, verify=False)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the JSON-decoded response

    def post_api(self, resource_path, payload=None, timeout=TIMEOUT, port=None, protocol=None, **kwargs):
        if self.__connected:
            print("\nChecking the Token Validity\n")
            self.__check_for_token_refresh()

        headers = {"Content-Type": "application/json", **self.common_headers}
        request_url = self.base_url + resource_path

        # log.info(f"\nPost API, Request URL: {request_url}")
        # log.info(f"\nPost API, Payload: {payload}")

        # print("\nPayload Information: ", payload)

        response = requests.post(request_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()  # Return the JSON-decoded response

    def connect(self, force=False):
        """ Generates a new ticket and establishes a fresh Rest client.
        Args:
            force (bool): If true, forces a new connection, else authenticates the existing one
        """

        if force:
            self.__connected = False

        self.__authentication()

    def disconnect(self):
        """ Disconnect from API client """

        print("Disconnecting the Rest client.")
        try:
            resource_path = "/api/system/" + self.version + "/identitymgmt/logoff"
            url = self.base_url + resource_path
            print("\nURL: {}\n".format(url))

            resource = requests.get(url, headers=self.common_headers, verify=False)
            log.info(f"Disconnect API Response: {resource.text}")
        except KeyError:
            print("Already disconnected Rest client.")

        self.__connected = False

    def __authentication(self):
        """ Generates a new authentication token for both oprim and converged. """

        if not self.__connected:
            resource_path = "/api/system/" + self.version + "/auth/token"
            url = self.base_url + resource_path
            # print("\nURL: {}\n".format(url))

            response = requests.post(url,
                                     auth=HTTPBasicAuth(self.username, self.password),
                                     verify=False
                                     )
            response.raise_for_status()
            response = response.json()

            # print("RestClientManager:\n Token Request Initiated At Time: '{}'".format(int(time.time())))
            # print("Response: ", response)

            if 'Token' in response:
                self._time_now = int(time.time())
                self.token_expiry_time = int(time.time()) + 3600
                self.token_refresh_time = self.token_expiry_time - 900
                headers = {"X-Auth-Token": response["Token"], "X-CSRF-Token": "soon-enabled"}
                self.common_headers = headers
            else:
                raise Exception("Cannot create REST client for an unauthorized user {}"
                                .format(self.username))

            self.__connected = True
        else:
            log.info("Already connected to Northbound API client.")

    def __check_for_token_refresh(self):
        """ Handles the token refresh """

        current_time = int(time.time())
        # log.info("RestClientManager:\n Current time: '{}'\n Token refresh time: '{}'\n Token expiry "
        #       "time:  '{}' \n".format(current_time, self.token_refresh_time, self.token_expiry_time))

        if current_time >= self.token_refresh_time:
            # if current_time >= self.token_expiry_time:
            #     log.info("Token Expired At Time: '{}'".format(self.token_expiry_time))
            #     log.info("Token Refresh is being initiated after token expiry time\n")
            # else:
            #     log.info("Token Expires At Time: '{}'".format(self.token_expiry_time))
            #     log.info("Token refresh is being initiated 15 minutes or less prior to the token expiry time\n")
            self.__connected = False
            self.__authentication()
        # else:
            # log.info("Token is still valid, token will be refreshed after: '{}' "
            #       "seconds.".format(self.token_refresh_time - current_time))
            # log.info("\nHeaders Information: {}".format(self.common_headers))

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
    print()
    if stored_data['device_ids'] or stored_data['site_ids']:
        if stored_data['device_ids']:
            # print("Stored Device IDs:")
            for idx, dev_id in enumerate(stored_data['device_ids'], start=1):
                print(f"{idx}. {dev_id}")
        else:
            print("No stored Device IDs.")

        if stored_data['site_ids']:
            # print("Stored Site IDs:")
            for idx, site_id in enumerate(stored_data['site_ids'], start=1):
                print(f"{idx}. {site_id}")
        else:
            print("No stored Site IDs.")
    else:
        print("No stored IDs available.")
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
        #print(f"Stored device ID: {device_id}")
    # Extract site ID
    site_id = item.get('siteId')
    if site_id and site_id not in stored_data['site_ids']:
        stored_data['site_ids'].append(site_id)
        #print(f"Stored site ID: {site_id}")

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
            max_tokens=2000,
            temperature=0.5,
            user=user_header_json
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        log.error(f"An error occurred while summarizing: {e}")
        return "No summary available."

# Helper functions for task management and credential retrieval

def check_task_and_get_credential_id(rest_client, task_id):
    """Checks task status and retrieves the credential ID."""
    url = f"/dna/intent/api/v1/task/{task_id}"
    while True:
        try:
            response = rest_client.get_api(url)
            task_response = response['response']
            if task_response.get('isError'):
                failure_reason = task_response.get('failureReason', '')
                print(f"Task failed: {failure_reason}")
                if 'already exists' in failure_reason.lower():
                    # Credential already exists
                    return None, 'ALREADY_EXISTS'
                else:
                    # Other error
                    return None, 'ERROR'
            else:
                progress = task_response.get('progress')
                if progress and 'credentialId' in progress:
                    # The progress field might contain JSON data as a string
                    progress_data = json.loads(progress.replace("'", "\""))
                    credential_id = progress_data.get('credentialId')
                    return credential_id, 'SUCCESS'
                elif 'endTime' in task_response:
                    # Task completed but no credentialId in progress
                    return None, 'COMPLETED_NO_CREDENTIAL_ID'
                else:
                    print("Task in progress...")
                    time.sleep(2)
        except Exception as e:
            print(f"Error checking task status: {e}")
            return None, 'ERROR'

def get_existing_credential_id(rest_client, credential_type, identifier):
    """Retrieves the existing credential ID based on type and identifier."""
    url = f"/dna/intent/api/v1/global-credential?credentialSubType={credential_type}"
    try:
        response = rest_client.get_api(url)
        for credential in response.get('response', []):
            if credential_type == 'CLI':
                # Match based on username
                if credential.get('username') == identifier:
                    return credential.get('id')
            elif credential_type in ['SNMPV2_READ_COMMUNITY', 'SNMPV2_WRITE_COMMUNITY']:
                # Match based on description
                if credential.get('description') == identifier:
                    return credential.get('id')
        return None
    except Exception as e:
        print(f"Error retrieving existing credential ID: {e}")
        return None

# Provisioning Workflow Functions

def provision_device_workflow(rest_client):
    print("Starting device provisioning workflow...")

    # Step 1: Configure CLI Credentials
    print("\nStep 1: Configure CLI Credentials")
    username = 'campus'
    password = 'Maglev123'
    enable_password = 'Maglev123'

    payload = [{
        "username": username,
        "password": password,
        "enablePassword": enable_password
    }]

    url = "/dna/intent/api/v1/global-credential/cli"
    try:
        response = rest_client.post_api(url, payload)
        task_id = response['response']['taskId']
        print(f"Credential creation initiated. Task ID: {task_id}")

        # Check task status and get credential ID
        credential_id, status = check_task_and_get_credential_id(rest_client, task_id)
        print(status)
        if status == 'SUCCESS' and credential_id:
            stored_data['cli_id'] = credential_id
            print(f"CLI Credential configured. ID: {credential_id}")
        elif status == 'ALREADY_EXISTS':
            print("CLI credential already exists. Retrieving existing credentials.")
            credential_id = get_existing_credential_id(rest_client, 'CLI', username)
            if credential_id:
                stored_data['cli_id'] = credential_id
                print(f"Using existing CLI Credential ID: {credential_id}")
            else:
                print("Failed to obtain existing CLI Credential ID.")
                return
        else:
            print("Failed to configure CLI Credential.")
            return
    except Exception as e:
        print(f"Error configuring CLI credentials: {e}")
        return

    # Step 2: Configure SNMP Read Community
    print("\nStep 2: Configure SNMP Read Community")
    read_community = 'campus'

    payload = [{
        "description": read_community,
        "readCommunity": read_community
    }]

    url = "/dna/intent/api/v1/global-credential/snmpv2-read-community"
    try:
        response = rest_client.post_api(url, payload)
        task_id = response['response']['taskId']
        print(f"SNMP Read Community creation initiated. Task ID: {task_id}")

        # Check task status and get credential ID
        credential_id, status = check_task_and_get_credential_id(rest_client, task_id)
        if status == 'SUCCESS' and credential_id:
            stored_data['snmp_read_id'] = credential_id
            print(f"SNMP Read Community configured. ID: {credential_id}")
        elif status == 'ALREADY_EXISTS':
            print("SNMP Read Community credential already exists. Retrieving existing credentials.")
            credential_id = get_existing_credential_id(rest_client, 'SNMPV2_READ_COMMUNITY', read_community)
            if credential_id:
                stored_data['snmp_read_id'] = credential_id
                print(f"Using existing SNMP Read Credential ID: {credential_id}")
            else:
                print("Failed to obtain existing SNMP Read Credential ID.")
                return
        else:
            print("Failed to configure SNMP Read Credential.")
            return
    except Exception as e:
        print(f"Error configuring SNMP read community: {e}")
        return

    # Step 3: Configure SNMP Write Community
    print("\nStep 3: Configure SNMP Write Community")
    write_community = 'campus'

    payload = [{
        "description": write_community,
        "writeCommunity": write_community
    }]

    url = "/dna/intent/api/v1/global-credential/snmpv2-write-community"
    try:
        response = rest_client.post_api(url, payload)
        task_id = response['response']['taskId']
        print(f"SNMP Write Community creation initiated. Task ID: {task_id}")

        # Check task status and get credential ID
        credential_id, status = check_task_and_get_credential_id(rest_client, task_id)
        if status == 'SUCCESS' and credential_id:
            stored_data['snmp_write_id'] = credential_id
            print(f"SNMP Write Community configured. ID: {credential_id}")
        elif status == 'ALREADY_EXISTS':
            print("SNMP Write Community credential already exists. Retrieving existing credentials.")
            credential_id = get_existing_credential_id(rest_client, 'SNMPV2_WRITE_COMMUNITY', write_community)
            if credential_id:
                stored_data['snmp_write_id'] = credential_id
                print(f"Using existing SNMP Write Credential ID: {credential_id}")
            else:
                print("Failed to obtain existing SNMP Write Credential ID.")
                return
        else:
            print("Failed to configure SNMP Write Credential.")
            return
    except Exception as e:
        print(f"Error configuring SNMP write community: {e}")
        return

    # Proceed with the rest of the provisioning workflow...

    # Step 4: Discover Switches
    print("\nStep 4: Discover Switches")
    discovery_name = input("Enter discovery name: ").strip()
    switch_list = input("Enter switch IP addresses (comma-separated): ").strip()

    payload = {
        "name": discovery_name,
        "discoveryType": "Multi range",
        "ipAddressList": switch_list,
        "globalCredentialIdList": [stored_data['cli_id'], stored_data['snmp_read_id'], stored_data['snmp_write_id']],
        "protocolOrder": "ssh,telnet",
        "snmpVersion": "v2",
        "timeout": "5",
        "retry": "3"
    }

    url = "/dna/intent/api/v1/discovery"
    try:
        response = rest_client.post_api(url, payload)
        task_id = response['response']['taskId']
        print(f"Discovery started. Task ID: {task_id}")
    except Exception as e:
        print(f"Error starting discovery: {e}")
        return

    # Step 5: Check Discovery Status
    print("\nStep 5: Checking Discovery Status")
    task_url = f"/dna/intent/api/v1/task/{task_id}"
    discovery_id = None

    # First, get the discovery ID from the task
    while True:
        try:
            response = rest_client.get_api(task_url)
            task_response = response.get('response', {})
            if isinstance(task_response, dict):
                is_error = task_response.get('isError', False)

                if is_error:
                    failure_reason = task_response.get('failureReason', 'Unknown error')
                    print(f"Discovery task failed: {failure_reason}")
                    return

                data_str = task_response.get('data', '')

                if data_str:
                    # Check if data_str is a JSON string
                    try:
                        data_json = json.loads(data_str)
                        # If data_json is a dict, extract discoveryId
                        if isinstance(data_json, dict):
                            discovery_id = data_json.get('discoveryId', '')
                    except json.JSONDecodeError:
                        # If data_str is not JSON, assume it's the discovery ID
                        discovery_id = data_str.strip('"')  # Remove any surrounding quotes

                    if discovery_id:
                        print(f"Retrieved Discovery ID: {discovery_id}")
                        break
                    else:
                        print(task_response)
                        print("Discovery ID not available yet. Waiting...")
                        time.sleep(5)
                else:
                    print("No 'data' field in task response. Waiting...")
                    time.sleep(5)
            else:
                print(f"Task response is not ready yet: {task_response}")
                print("Waiting...")
                time.sleep(5)
        except Exception as e:
            print(f"Error retrieving discovery ID: {e}")
            return

    # Now poll the discovery status using the discovery ID
    print("\nPolling Discovery Status")
    discovery_url = f"/dna/intent/api/v1/discovery/{discovery_id}"

    while True:
        try:
            response = rest_client.get_api(discovery_url)
            discovery_response = response.get('response', {})
            if isinstance(discovery_response, dict) and len(discovery_response) > 0:

                status = discovery_response.get('discoveryStatus', '')
                if status.lower() == 'inactive':
                    print("Discovery completed successfully.")
                    print("Sleeping to allow for device inventory...")
                    time.sleep(60)

                    break
                elif status.lower() == 'error':
                    print("Discovery failed.")
                    return
                else:
                    print(f"Discovery status: {status}")
                    print("Discovery in progress...")
                    time.sleep(5)
            else:
                print(discovery_response)
                print("Discovery information not available yet. Waiting...")
                time.sleep(5)
        except Exception as e:
            print(f"Error polling discovery status: {e}")
            return

    # Step 6: Get Discovered Switches
    print("\nStep 6: Retrieving Discovered Switches")
    url = "/dna/intent/api/v1/network-device/"
    try:
        response = rest_client.get_api(url)
        switch_dict = {}
        for switch in response.get("response", []):
            hostname = switch.get('hostname')
            device_id = switch.get('id')
            if hostname and device_id:
                switch_dict[hostname] = device_id
        if not switch_dict:
            print("No switches discovered.")
            return
        print("Discovered Switches:")
        for idx, (hostname, device_id) in enumerate(switch_dict.items(), start=1):
            print(f"{idx}. Hostname: {hostname}, Device ID: {device_id}")
    except Exception as e:
        print(f"Error retrieving discovered switches: {e}")
        return

    # Step 7: Get Sites
    print("\nStep 7: Retrieving Sites")
    url = "/dna/intent/api/v1/site"
    try:
        response = rest_client.get_api(url)
        site_dict = {}
        for site in response.get("response", []):
            site_name = site.get('siteNameHierarchy')
            site_id = site.get('id')
            if site_name and site_id:
                site_dict[site_name] = site_id
        if not site_dict:
            print("No sites available.")
            return
        print("Available Sites:")
        for idx, (site_name, site_id) in enumerate(site_dict.items(), start=1):
            print(f"{idx}. Site: {site_name}, Site ID: {site_id}")
    except Exception as e:
        print(f"Error retrieving sites: {e}")
        return

    # Step 8: Provision Site
    print("\nStep 8: Provisioning Site")
    # Select a site
    site_selection = input("Enter the site name you want to provision to: ").strip()
    site_id = site_dict.get(site_selection)
    if not site_id:
        print("Invalid site selection.")
        return

    # Select a switch
    switch_selection = input("Enter the hostname of the switch to provision: ").strip()
    device_id = switch_dict.get(switch_selection)
    if not device_id:
        print("Invalid switch selection.")
        return

    payload = [
        {
            "siteId": site_id,
            "networkDeviceId": device_id
        }
    ]

    url = "/dna/intent/api/v1/sda/provisionDevices"
    try:
        response = rest_client.post_api(url, payload)
        task_id = response['response']['taskId']
        print(f"Provisioning started. Task ID: {task_id}")
    except Exception as e:
        print(f"Error starting provisioning: {e}")
        return

    # Step 9: Check Provisioning Status
    print("\nStep 9: Checking Provisioning Status")
    url = f"/dna/intent/api/v1/task/{task_id}"
    while True:
        try:
            response = rest_client.get_api(url)
            progress = response['response'].get('data', '')
            if not progress:
                print("No progress data available yet. Provisioning in progress...")
                time.sleep(5)
                continue
            try:
                # Parsing the 'progress' data string
                data_pairs = progress.split(';')
                data_dict = {}
                for pair in data_pairs:
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        data_dict[key.strip()] = value.strip()

                processcfs_complete = data_dict.get('processcfs_complete', '').lower()
                failure_task = data_dict.get('failure_task', 'NA').lower()

                if processcfs_complete == 'true':
                    print("Provisioning completed successfully.")
                    break
                elif failure_task != 'na':
                    print(f"Provisioning failed: {failure_task}")
                    return
                else:
                    print("Provisioning in progress...")
                    time.sleep(5)
            except Exception as e:
                print(f"Error parsing provisioning status: {e}")
                return
        except Exception as e:
            print(f"Error checking provisioning status: {e}")
            return

    print("Device provisioning workflow completed successfully.")


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
        # Check if the matched question is for provisioning
        if best_question == "provision a device":
            provision_device_workflow(rest_client)
        else:
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
        "provision a device": None,  # We'll handle this in the hybrid_search function
        "get devices assigned to a site": {
            "path": "/dna/intent/api/v1/site-member/{id}/member",
            "method": "get",
            "operation_id": "getDevicesThatAreAssignedToASite",
            "summary": "Get devices that are assigned to a site",
            "description": "API to get devices that are assigned to a site.",
            "tags": ["Sites"]
        },
        "what are the devices I have": {
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
