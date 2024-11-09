import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('TOGETHER_AI_API_KEY')

# TogetherAI API endpoint
url = "https://api.together.xyz/inference"

# Test prompt
prompt = "Translate the following English text to French: 'Hello, how are you?'"

# Request payload
payload = {
    "model": "togethercomputer/llama-2-7b-chat",
    "prompt": prompt,
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.7,
    "top_k": 50,
    "repetition_penalty": 1
}

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Send the request
response = requests.post(url, json=payload, headers=headers)

# Check the response
if response.status_code == 200:
    result = response.json()
    print("API connection successful!")
    print("Generated text:", result['output']['choices'][0]['text'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
