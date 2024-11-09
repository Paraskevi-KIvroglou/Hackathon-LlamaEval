import os
import requests
import together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('TOGETHER_AI_API_KEY')

# TogetherAI API endpoint
url = "https://api.together.xyz/inference"

# Test prompt
prompt = "Translate the following English text to French: 'Hello, how are you?'"


client = together.Together(api_key=api_key)

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
)
print(response.choices[0].message.content)

# Headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}