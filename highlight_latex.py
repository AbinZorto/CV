import json
import requests
import re
import time
import shutil

def execute_workflow(text_input):
    # API configuration
    headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMjQyYmI1N2UtOWFhZC00ZTc0LWI0Y2QtZTEzMjVhYjU1NmUzIiwidHlwZSI6ImFwaV90b2tlbiJ9.IwYXJembfCpHk6td9bBTzI-X6HeewsuTTEffGzZZmk4"}
    base_url = "https://api.edenai.run/v2/workflow/5b8c97b1-9b91-4c87-a22c-224f0d23928b"

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
            time.sleep(10)
        else:
            raise Exception(f"Unknown status: {current_status}")
    
    raise Exception("Workflow timed out")

def highlight_latex_file(file_path, api_result):
    """Highlight AI-detected text in LaTeX file based on API results."""
    try:
        # Read the original file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Create backup
        backup_path = file_path + '.backup'
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at: {backup_path}")

        # Get the items from the API result
        items = api_result.get('text__ai_detection', {}).get('results', [{}])[0].get('items', [])
        
        # Add required packages if they don't exist
        if r"\usepackage{soul}" not in content:
            packages = (
                r"\usepackage{xcolor}" + '\n' +
                r"\usepackage{soul}" + '\n' +
                r"\sethlcolor{yellow!30}" + '\n' +
                r"\newcommand{\AIHighlight}[1]{\hl{#1}}" + '\n'
            )
            # Find documentclass line
            match = re.search(r'\\documentclass.*?\n', content)
            if match:
                insert_pos = match.end()
                content = content[:insert_pos] + packages + content[insert_pos:]
                print("Added required LaTeX packages")

        modified_content = content
        highlight_count = 0

        # Process AI-detected segments
        for item in items:
            if item.get('prediction') == 'ai-generated' and item.get('ai_score', 0) > 0.6:
                text = item['text'].strip()
                score = item['ai_score']
                
                # Clean and escape the text for regex
                clean_text = text.replace('\\', '\\\\')
                clean_text = re.escape(clean_text)
                
                # Only highlight if not already highlighted
                if not re.search(r'\\AIHighlight{[^}]*' + clean_text + r'[^}]*}', modified_content):
                    try:
                        # Replace the text with highlighted version
                        pattern = f"(?<!\\\\AIHighlight{{)({clean_text})"
                        replacement = r"\\AIHighlight{\1}"
                        new_content = re.sub(pattern, replacement, modified_content)
                        
                        if new_content != modified_content:
                            modified_content = new_content
                            highlight_count += 1
                            print(f"Highlighted text with AI score {score:.2f}")
                    except Exception as e:
                        print(f"Warning: Could not highlight text: {text[:50]}...")
                        print(f"Error: {str(e)}")

        # Write modified content back to file
        if highlight_count > 0:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(modified_content)
            print(f"\nHighlighted {highlight_count} AI-detected segments")
        else:
            print("\nNo changes were made to the file")

    except Exception as e:
        print(f"Error in highlight_latex_file: {str(e)}")
        raise

# Example usage
try:
    result = execute_workflow("ABIN ZORTOMACHINE LEARNING ENGINEERFlat 6, 60 West-ferry Road, London, E148LN — +447939412710 —abinzorto@outlook.com — ¯linkedin.com/in/abin-zortoOctober 27, 2024Hiring ManagerComplexioFoundational AIDear Hiring Manager,I am writing to express my strong interest in the DevOps Engineer position at Complexio.With my experience in deploying andmanaging ML pipelines across multiple cloud environments and my extensive work with Kubernetes, I believe I can contributesignificantly to Complexio’s mission of automating business activities through AI solutions.Currently pursuing my PhD in Computer Science at the University of East London, I have gained valuable hands-on experience inmanaging distributed systems and implementing ML solutions at scale.As a Senior Research Assistant, I have:• Architected and managed multi-cloud ML pipelines across AWS, Azure, and Google Cloud, processing 100,000+ datapoints daily• Successfully deployed and managed Kubernetes clusters for ML workloads, improving system reliability by 35%• Implemented robust CI/CD pipelines using GitHub Actions and Jenkins, reducing deployment time by 50%• Worked extensively with Neo4j and vector databases for efficient data management and similarity search• Developed and optimized data integration pipelines connecting multiple sources to centralized data lakesI am particularly excited about Complexio’s focus on leveraging multiple LLMs and graph databases to understand and automatebusiness processes.My experience in managing GPU-accelerated ML workflows and optimizing distributed systems aligns perfectlywith your technical requirements.Additionally, my background in implementing secure data integration workflows compliant withhealthcare regulations demonstrates my understanding of handling sensitive data at scale.While I may not meet the exact experience requirements of 7+ years in cloud infrastructure, my intensive hands-on experiencewith cutting-edge technologies and proven track record of successfully implementing complex ML systems demonstrates mycapability to handle the responsibilities of this role effectively.I am confident that my technical expertise, coupled with my research background in AI and distributed systems, would make me avaluable addition to your team.I would welcome the opportunity to discuss how my skills and experience align with Complexio’sneeds in more detail.Thank you for considering my application.Best regards,Abin Zorto1")
    print("\nWorkflow result:", json.dumps(result.get('content', {}).get('results', {}), indent=2))
except Exception as e:
    print(f"Error: {e}")
