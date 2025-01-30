import os
import pandas as pd
import time
import random
from httpx import HTTPStatusError
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.output_parsers import StrOutputParser


# Load environment variables and prepare balanced dataset for sentiment analysis
load_dotenv()
MISTRAL_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_PATH = os.getenv("DATA_PATH")

data_path = os.getenv("DATA_PATH")
data = pd.read_csv(data_path)
data = data[data['description'].notna()]

min_count = data["star"].value_counts().min()
balanced_df = data.groupby("star").apply(lambda x: x.sample(min_count)).reset_index(drop=True)

balanced_df_comments = balanced_df[(balanced_df['description'].notna())][['description', 'star']]
comments = balanced_df_comments[balanced_df_comments['description'].apply(lambda x: len(x.split(' '))) > 5].sample(50000)
print(1)
# Initialize Mistral AI model and parser
model = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_KEY)
parser = StrOutputParser()

# Template for single sentiment classification
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
print(2)

prompt1 = ChatPromptTemplate.from_template(template_for_sentiment)
chain1 = prompt1 | model | parser

prompt_batch = ChatPromptTemplate.from_template(template_for_sentiment_batch)
chain_batch = prompt_batch | model | parser

# Handles single comment classification with exponential backoff retry mechanism
def classify_with_backoff(comment, max_retries=5):
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



# Process sample comments for testing
if 'description' in comments and not comments['description'].empty:
    for i in comments['description'].sample(min(5, len(comments['description']))).index:
        sentiment = classify_with_backoff(comments['description'].loc[i])
        print(f"Sentiment: {sentiment}")
        print(f"Comment: {comments['description'].loc[i]}")
        print(f"star: {comments['star'].loc[i]} \n\n")
else:
    print("No valid comments found.")


# Batch processing configuration and execution
BATCH_SIZE = 10

# Check if 'description' column exists
if 'description' in comments and not comments['description'].empty:
    comments_list = comments['description'].tolist()
    sentiment_results = []

    # Process in batches
    for i in range(0, len(comments_list), BATCH_SIZE):
        batch = comments_list[i:i + BATCH_SIZE]
        sentiments = classify_batch_with_backoff(batch)
        sentiment_results.extend(sentiments)

    # Add results to DataFrame
    comments['sentiment'] = sentiment_results
else:
    print("No valid comments found.")

# Print a sample of the updated DataFrame
print(comments.head())
comments.to_csv("comments_with_sentiment.csv", index=False)