import os
import time
import random
from httpx import HTTPStatusError
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser


# Load environment variables and prepare balanced dataset for sentiment analysis
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize Mistral AI model and parser
model = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
parser = StrOutputParser()

# Template for single sentiment classification
# Note: Examples are in Persian/Farsi language for sentiment analysis
template_for_sentiment = """
You are an expert sentiment classifier. Classify the sentence below as "Positive", "Negative", or "Neutral",
based on the examples provided.
Just answer in one word.

Examples:
sentence: "عالی بود"
Sentiment: Positive

sentence: "سلام خیلی دیر به دستم رسید"
Sentiment: Negative

sentence: "معمولی بود"
Sentiment: Neutral

sentence: "جمع و جور و خوش دست هست"
Sentiment: Positive

Now, classify this new sentence:
"{sentence}"
"""

# Template for batch sentiment classification
template_for_sentiment_batch = """
You are an expert sentiment classifier. Classify each sentence below as "Positive", "Negative", or "Neutral",
based on the examples provided.
Just answer in one word per sentence.

Examples:
sentence: "عالی بود"
Sentiment: Positive

sentence: "سلام خیلی دیر به دستم رسید"
Sentiment: Negative

sentence: "معمولی بود"
Sentiment: Neutral

sentence: "جمع و جور و خوش دست هست"
Sentiment: Positive

Now, classify these new sentences:
{sentences}

Return only the sentiment labels in the same order, separated by new lines.
"""

prompt1 = ChatPromptTemplate.from_template(template_for_sentiment)
chain1 = prompt1 | model | parser

prompt_batch = ChatPromptTemplate.from_template(template_for_sentiment_batch)
chain_batch = prompt_batch | model | parser

# Handles single comment classification with exponential backoff retry mechanism
def classify_with_backoff(comment, max_retries=5):
    """
    Classifies a single comment with retry mechanism for rate limiting
    Args:
        comment (str): The text to classify
        max_retries (int): Maximum number of retry attempts
    Returns:
        str: Sentiment classification result or "Error"
    """
    retries = 0
    while retries < max_retries:
        try:
            response = chain1.invoke({"sentence": comment})
            return response.strip()
        except HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** retries + random.uniform(0, 1)
                print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise
    return "Error"

# Handles batch classification with exponential backoff retry mechanism
def classify_batch_with_backoff(sentences, max_retries=5):
    """
    Classifies multiple sentences in batch with retry mechanism
    Args:
        sentences (list): List of texts to classify
        max_retries (int): Maximum number of retry attempts
    Returns:
        list: List of sentiment classifications or "Error" for each input
    """
    retries = 0
    while retries < max_retries:
        try:
            response = chain_batch.invoke({"sentences": "\n".join([f"sentence: \"{s}\"" for s in sentences])})
            return response.strip().split("\n")
        except HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = 2 ** retries + random.uniform(0, 1)
                print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                raise
    return ["Error"] * len(sentences)


def comment_classification(comments, method="single", max_retries=5):
    """
    Main classification function that handles both single and batch processing
    Args:
        comments (str or list): Input text(s) to classify
        method (str): "single" for one comment, "batch" for multiple comments
        max_retries (int): Maximum number of retry attempts
    Returns:
        str or list: Sentiment classification result(s)
    """
    if method == "single":
        return classify_with_backoff(comments, max_retries) 
    elif method == "batch":
        return classify_batch_with_backoff(comments, max_retries)
    else:
        raise ValueError(f"Invalid method: {method}")

if __name__ == "__main__":
    print(comment_classification("عالی بود", method="single"))
    print(comment_classification(["عالی بود", "سلام خیلی دیر به دستم رسید", "معمولی بود", "جمع و جور و خوش دست هست"], method="batch"))
