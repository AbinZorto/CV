# ai_check.sh
#!/bin/bash

# Extract text from PDF
pdftotext "$1" output.txt

# Check if output.txt exists and has content
if [ ! -s output.txt ]; then
    echo "Text extraction failed or PDF is empty."
    exit 1
fi

# Send the extracted text to an AI detection API
# Replace API_ENDPOINT and API_KEY with the AI detector's details

API_ENDPOINT="https://www.freedetector.ai/api/content_detector/"  # Placeholder URL
API_KEY="243c8b4a8e6a7ee49a13a77cd78ff8fb"  # Replace with your API key

response=$(curl -s -X POST "$API_ENDPOINT" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: text/plain" \
    --data-binary "@output.txt")

# Print the response from the API
echo "AI Detection Result: $response"
