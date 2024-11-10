def get_descriptions():
    dictionary = {
        "stanfordnlp/imdb" : "Large Movie Review Dataset. This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well.",
        "rajpurkar/squad" : "Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. " +
                            "SQuAD 1.1 contains 100,000+ question-answer pairs on 500+ articles.",
        "abisee/cnn_dailymail" : "The CNN / DailyMail Dataset is an English-language dataset containing just over 300k unique news articles as written by journalists at CNN and the Daily Mail. The current version supports both extractive and abstractive summarization, though the original version was created for machine reading and comprehension and abstractive question answering. ",
                
    }
    return dictionary

def get_datasets():
    dictionary = {
        "qa" : "rajpurkar/squad",
        "sentiment" : "stanfordnlp/imdb",
        "classification" : "stanfordnlp/imdb",
        "summarization" : "abisee/cnn_dailymail" 
    }
    return dictionary