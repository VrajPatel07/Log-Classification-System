from llm_classifier import classify_with_llm
from bert_classifier import classify_with_bert
from regex_classifier import classify_with_regex
import pandas as pd



def classify_log(source, log_msg):
    """
        task : classify one log
        input : error source and log message
        output : error category
    """
    # use LLM if source == "LegacyCRM"
    if source == "LegacyCRM":
        label = classify_with_llm(log_msg)
    # if not then use regex or BERT
    else:
        # use regex first
        label = classify_with_regex(log_msg)
        # use BERT if regex return None
        if not label:
            label = classify_with_bert(log_msg)
    # return error categoty
    return label


def classify(logs):
    """
        task : classify multiple logs
        input : array of logs - {source, log_message}
        output : category for all input logs
    """
    labels = []
    for source, log_msg in logs:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels



def classify_csv(input_file):
    """
        task : classify all logs given in the CSV file
        input : CSV file path
        output : CSV file with added "target_label" column
    """
    # load CSV file
    df = pd.read_csv(input_file)
    # Perform classification
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))
    # Save the modified file
    output_file_path = "testing/output.csv"
    df.to_csv(output_file_path, index=False)
    # return output file
    return output_file_path