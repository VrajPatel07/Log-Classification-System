import joblib
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-mpnet-base-v2"

model_embedding = SentenceTransformer(model_name)
model_classification = joblib.load("models/log_classifier.joblib")


def classify_with_bert(log_message):
    """
        input : log message
        output : error type classified using BERT
    """
    # generate embedding for given log message
    embeddings = model_embedding.encode([log_message])
    # calculate probability of log message being of each class
    probabilities = model_classification.predict_proba(embeddings)[0]
    # return "Unclassified" if max probability < 0.5
    if max(probabilities) < 0.5:
        return "Unclassified"
    # return classified label
    predicted_label = model_classification.predict(embeddings)[0]
    
    return predicted_label