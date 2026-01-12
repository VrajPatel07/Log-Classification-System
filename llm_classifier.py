from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv


load_dotenv()


# Pydantic schema
class LogClassification(BaseModel):
    category : Literal[
        "Workflow Error",
        "Deprecation Warning",
        "Unclassified"
    ] = Field(description = "Predicted category of the log message.")



# Gemini model
model = genai.Client()



def classify_with_llm(log_message):
    # prompt
    prompt = f"""
    Classify the log message into one of these categories:
    (1) Workflow Error
    (2) Deprecation Warning

    If you can't figure out a category, use "Unclassified".

    Return ONLY valid JSON that matches the schema.
    Log message : {log_message}
    """
    # generate response
    response = model.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt,
        config = {
            "response_mime_type" : "application/json",
            "response_json_schema" : LogClassification.model_json_schema()
        }
    )
    # return categoty
    classification  = LogClassification.model_validate_json(response.text)
    return classification.category