def get_Llama_Models() -> dict:
    dictionary = {
        # "Meta Llama 3.2 90B Vision Instruct Turbo" : "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", 
        # "Meta Llama 3.2 11B Vision Instruct Turbo" : "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "Meta Llama 3.2 3B Instruct Turbo" : "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "Meta Llama 3.1 8B Instruct Turbo" : "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Meta Llama 3.1 70B Instruct Turbo" : "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "Meta Llama 3 70B Instruct Lite" : "meta-llama/Meta-Llama-3-70B-Instruct-Lite",
        "Meta Llama 3 8B Instruct Lite" : "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
    }
    return dictionary

def get_evaluation_tasks() -> dict:
    dictionary = {
        "Quantative Analysis" : "qa", 
        "Sentiment Analysis" : "sentiment",
        "Classification" : "classification",
        "Summarization" : "summarization"
    }
    return dictionary

def get_metrics_to_print(task, results) -> dict:
    dictionary = { }
    if task.lower() == 'qa':
        dictionary["Exact Match Score"] = f"{results[0]}"
        dictionary["F1 Score"] = f"{results[1]:.2f}"

    elif task.lower() == 'summarization':
        dictionary["BLEU Score"] = f"{results[0]:.2f}"
        dictionary["ROUGE-1 Score F1"] = f"{results[1]:.2f}"
        dictionary["ROUGE-1 Score Precision"] = f"{results[2]:.2f}"
        dictionary["ROUGE-1 Score Recall"] = f"{results[3]:.2f}"
        dictionary["ROUGE-2 Score F1"] = f"{results[4]:.2f}"
        dictionary["ROUGE-2 Score Precision"] = f"{results[5]:.2f}"
        dictionary["ROUGE-2 Score Recall"] = f"{results[6]:.2f}"
        dictionary["ROUGE-L Score F1"] = f"{results[7]:.2f}"
        dictionary["ROUGE-L Score Precision"] = f"{results[8]:.2f}"
        dictionary["ROUGE-L Score Recall"] = f"{results[9]:.2f}"

    elif task.lower() == 'sentiment' or task.lower() == 'classification':
        dictionary["Accuracy"] = f"{results[0]:.2f}"
    return dictionary