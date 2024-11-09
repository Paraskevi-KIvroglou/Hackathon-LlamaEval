import os
import requests
import together
import json, time
from dotenv import load_dotenv

import test_benchmarks as benchmarks
import fetch_dataset as fetch
import datasets

import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

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
    return response.choices[0].message.content

def call_API_request(model, prompt):
    url = "https://api.together.xyz/inference"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "prompt" : prompt,
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        print(f"Response content: {response.text}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        print(f"Response content: {response.text}")
        return None
    
def pretty_print_json(json_object):
    # Convert the JSON object to a formatted string with indentation
    formatted_json = json.dumps(json_object, indent=4, sort_keys=True)
    
    # Print the formatted JSON string
    print(formatted_json)

def run_batches(dataset, model):
# Process items in smaller batches
    batch_size = 100  # Adjust this based on your needs
    results = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset.iloc[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced max_workers
            futures = [executor.submit(call_API_request, model, f"{row['context']} {row['question']}") for _, row in batch.iterrows()]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    print("Skipping None result")
        
        print(f"Processed batch {i//batch_size + 1}/{len(dataset)//batch_size + 1}")
        time.sleep(1)  # Add a small delay between batches

    # Process results as needed
    for result in results:
        print(pretty_print_json(result['output']['choices'][0]['text']))
    return results


def evaluate_response(response, reference, task):
    evaluation = benchmarks.evaluate_model(response, reference, task)
    return evaluation

def previous_main(prompt, expected):
    response = call_API_from_client("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", prompt=prompt)
    evaluation = evaluate_response(response=response, reference=expected,task="summarization")
    print(evaluation)

def main():
    dataset, json_dataset = fetch.fetch_dataset('qa')

    results = run_batches(dataset=dataset, model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

    return results

main()