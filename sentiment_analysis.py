import os
import time
import random
from httpx import HTTPStatusError
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize the SentimentAnalyzer class.
        Loads environment variables, initializes the model and parser, and defines templates.
        """
        # Load environment variables
        load_dotenv()
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        
        # Initialize model and parser
        self.model = ChatMistralAI(model="mistral-large-latest", api_key=self.mistral_api_key)
        self.parser = StrOutputParser()
        
        # Define templates
        self._template_for_sentiment = """
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

        self._template_for_sentiment_batch = """
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

        self._initialize_chains()

    def _initialize_chains(self):
        """
        Initialize the prompt chains for single and batch sentiment analysis.
        """
        self.prompt_single = ChatPromptTemplate.from_template(self._template_for_sentiment)
        self.chain_single = self.prompt_single | self.model | self.parser

        self.prompt_batch = ChatPromptTemplate.from_template(self._template_for_sentiment_batch)
        self.chain_batch = self.prompt_batch | self.model | self.parser

    def update_templates(self, single_template=None, batch_template=None):
        """
        Update the templates used for sentiment analysis.
        
        Args:
            single_template (str, optional): New template for single comment analysis.
            batch_template (str, optional): New template for batch analysis.
        """
        if single_template is not None:
            self._template_for_sentiment = single_template
        if batch_template is not None:
            self._template_for_sentiment_batch = batch_template
        
        # Reinitialize the chains with new templates
        self._initialize_chains()

    def _classify_with_backoff(self, comment, max_retries=5):
        """
        Classifies a single comment with a retry mechanism for rate limiting.
        
        Args:
            comment (str): The comment to classify.
            max_retries (int): Maximum number of retry attempts.
        
        Returns:
            str: The sentiment classification result.
        """
        retries = 0
        while retries < max_retries:
            try:
                response = self.chain_single.invoke({"sentence": comment})
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

    def _classify_batch_with_backoff(self, sentences, max_retries=5):
        """
        Classifies multiple sentences in batch with a retry mechanism.
        
        Args:
            sentences (list): List of sentences to classify.
            max_retries (int): Maximum number of retry attempts.
        
        Returns:
            list: List of sentiment classification results.
        """
        retries = 0
        while retries < max_retries:
            try:
                response = self.chain_batch.invoke({"sentences": "\n".join([f"sentence: \"{s}\"" for s in sentences])})
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

    def classify(self, comments, method="single", max_retries=5):
        """
        Main classification method that handles both single and batch processing.
        
        Args:
            comments (str or list): Input text(s) to classify.
            method (str): "single" for one comment, "batch" for multiple comments.
            max_retries (int): Maximum number of retry attempts.
        
        Returns:
            str or list: Sentiment classification result(s).
        """
        if method == "single":
            return self._classify_with_backoff(comments, max_retries)
        elif method == "batch":
            return self._classify_batch_with_backoff(comments, max_retries)
        else:
            raise ValueError(f"Invalid method: {method}")

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    print(analyzer.classify("عالی بود", method="single"))
    
    print(analyzer.classify(
        ["عالی بود", "سلام خیلی دیر به دستم رسید", "معمولی بود", "جمع و جور و خوش دست هست"],
        method="batch"
    ))

    print(analyzer.classify('راننده موتورش خیلی بد بود', method='single'))
    print('test new tamplate')
    # Update single template example
    new_single_template = """
    You are an expert sentiment classifier. Classify each sentence below as "True", 'Flase,
    return true if sentence is about delivery, based on the examples provided.
    Just answer in one word per sentence.

    Examples:
    sentence: "خیلی دیر به دستم رسید"
    Sentiment: True

    sentence: "خیلی خوش مزه بود"
    Sentiment: False

    sentence: "سرویسش خیلی بد بود"
    Sentiment: True

    Now, classify this new sentence:
    Text: "{sentence}"
    """

    analyzer.update_templates(single_template=new_single_template)
    print(analyzer.classify('راننده موتورش خیلی بد بود', method='single'))

