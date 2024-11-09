from datasets import load_dataset
import pandas as pd

def process_sentiment_dataset(ds_name, size):

    data = load_dataset(ds_name[:size]).shuffle(seed=42) 
    df = pd.DataFrame(data['train'])
    print(df.head())

    print(df['label'].value_counts())

    label2id = {'negative': 0, 'positive': 1}
    id2label = {0: 'negative', 1: 'positive'}

    df['label'] = df['label'].apply(lambda x: id2label[x])  # dataset library creates this new column. iterates over each row

    # Check the DataFrame after mapping
    print(df.head())
    print((df['label']=='positive').value_counts())

    return df

def process_qa_dataset(ds_name, size):

    data = load_dataset(ds_name[:size]).shuffle(seed=42) 
    df = pd.DataFrame(data['train'])
    print(df.head())

    return df

def process_summarization_dataset(ds_name, ds_version, size):

    data = load_dataset(ds_name[:size], ds_version).shuffle(seed=42) 
    df = pd.DataFrame(data['train'])
    print(df.head())

    return df

def fetch_dataset(task_type, size = 50):
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

# fetch_dataset('sentiment')
# fetch_dataset('qa')
# fetch_dataset('classification')
# fetch_dataset('summarization')
