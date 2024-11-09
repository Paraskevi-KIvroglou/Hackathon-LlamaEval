import os
import requests
import together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('TOGETHER_AI_API_KEY')
client = together.Together(api_key=api_key)

def call_API_from_client(model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    print(response.choices[0].message.content)

