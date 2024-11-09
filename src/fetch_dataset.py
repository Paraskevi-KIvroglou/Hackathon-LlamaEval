from datasets import load_dataset
import pandas as pd
import json

dataset_size = 100

def process_sentiment_dataset(ds_name, size):
    # Load and shuffle dataset
    data = load_dataset(ds_name).shuffle(seed=42)
    df = pd.DataFrame(data['test'][:size])

    # Display initial info
    # print(df.head())
    # print(df['label'].value_counts())

    # Mapping labels
    label2id = {'negative': 0, 'positive': 1}
    id2label = {0: 'negative', 1: 'positive'}
    df['label'] = df['label'].apply(lambda x: id2label[x])

    # Display processed data
    # print(df.head())
    # print((df['label'] == 'positive').value_counts())

    # Convert DataFrame to JSON format
    result = df.to_dict(orient='records')
    return df, json.dumps(result, indent=4)  # Convert to JSON string with pretty-printing

def process_qa_dataset(ds_name, size):
    # Load and shuffle dataset
    data = load_dataset(ds_name).shuffle(seed=42)
    df = pd.DataFrame(data['validation'][:size])

    # Display initial info
    #print(df.head())

    # Convert DataFrame to JSON format
    result = df.to_dict(orient='records')
    return df, json.dumps(result, indent=4)

def process_summarization_dataset(ds_name, ds_version, size):
    # Load and shuffle dataset
    data = load_dataset(ds_name, ds_version).shuffle(seed=42)
    df = pd.DataFrame(data['test'][:size])

    # Display initial info
    #print(df.head())

    # Convert DataFrame to JSON format
    result = df.to_dict(orient='records')
    return df, json.dumps(result, indent=4)

def fetch_dataset(task_type, size = dataset_size):
    try:
        if task_type.lower() == 'qa':
            ds_name = "squad"  #TODO : trivia_qa, natural_questions, quoref
            return process_qa_dataset(ds_name, size)
        if task_type.lower() == 'classification' or task_type.lower() == 'sentiment':
            ds_name = "imdb" #TODO sst2, yelp_polarity
            return process_sentiment_dataset(ds_name, size)
        if task_type.lower() == 'summarization':
            ds_name = "cnn_dailymail" #TODO xsum, gigaword
            cnn_ds_version = '3.0.0'
            return process_summarization_dataset(ds_name, cnn_ds_version, size)
    except:
        print(f"Error extracting dataset {ds_name}. Please try another.")
        raise

#js_1 = fetch_dataset('sentiment')
#js_1 = fetch_dataset('qa')
# js_1 = fetch_dataset('classification')
# js_1 = fetch_dataset('summarization')

#print(js_1)
