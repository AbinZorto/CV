import json
import requests
import time

def execute_workflow(text_input):
    # API configuration
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMjQyYmI1N2UtOWFhZC00ZTc0LWI0Y2QtZTEzMjVhYjU1NmUzIiwidHlwZSI6ImFwaV90b2tlbiJ9.IwYXJembfCpHk6td9bBTzI-X6HeewsuTTEffGzZZmk4"}
    base_url = "https://api.edenai.run/v2/workflow/0da2c342-cc49-4507-8d38-fa9c0892115b"


    # Start workflow execution
    payload = {"text": text_input}
    response = requests.post(f"{base_url}/execution/", json=payload, headers=headers)
    
    # Accept both 200 and 201 as success codes
    if response.status_code not in [200, 201]:
        raise Exception(f"Error: Status code {response.status_code}")
    
    initial_result = response.json()
    execution_id = initial_result.get('id')  # Changed from 'execution_id' to 'id'
    
    if not execution_id:
        raise Exception("No execution ID received")

    print(f"Workflow started with execution ID: {execution_id}")

    # Check workflow status
    max_attempts = 10  # Reduced number of attempts
    attempts = 0

    while attempts < max_attempts:
        status_response = requests.get(f"{base_url}/execution/{execution_id}/", headers=headers)
        status_result = status_response.json()
        current_status = status_result.get('content', {}).get('status')  # Updated path to status
        
        print(f"Current status: {current_status}")
        
        if current_status == 'success':  # Changed from 'finished' to 'success'
            print("Workflow completed successfully!")
            return status_result
        elif current_status in ['failed', 'error']:
            raise Exception("Workflow execution failed")
        elif current_status == 'running':
            attempts += 1
            time.sleep(2)
        else:
            raise Exception(f"Unknown status: {current_status}")
    
    raise Exception("Workflow timed out")

# Example usage
try:
    result = execute_workflow("Test message")
    print("\nWorkflow result:", json.dumps(result.get('content', {}).get('results', {}), indent=2))
except Exception as e:
    print(f"Error: {e}")
