# Sentiment Analysis with LLM API

This repository contains a sentiment analysis implementation using the Mistral AI API through LangChain. It's specifically designed to analyze Persian/Farsi text comments and can classify sentiments as "Positive", "Negative", or "Neutral". The implementation supports:

1. **Single API Call per Comment** - Individual comment analysis
2. **Batch API Calls** - Multiple comments processed in a single API request
3. **Customizable Templates** - Ability to modify classification templates for different use cases

## Setup

### Prerequisites
- Python 3.8+
- Required packages: langchain, langchain-mistralai, python-dotenv, httpx
- Mistral AI API key

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Set up your API key:
   - Create a `.env` file in the root directory
   - Add your Mistral AI API key:
     ```sh
     MISTRAL_API_KEY=your_api_key_here
     ```

## Usage

### Basic Usage
```python
from sentiment_analysis import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Single comment analysis
result = analyzer.classify("عالی بود", method="single")

# Batch analysis
comments = ["عالی بود", "سلام خیلی دیر به دستم رسید", "معمولی بود"]
results = analyzer.classify(comments, method="batch")
```

### Customizing Templates
You can customize the analysis templates for different use cases:

```python
# Example: Changing to a delivery-focused classification
new_template = """
You are an expert sentiment classifier. Classify each sentence below as "True", "False",
return true if sentence is about delivery, based on the examples provided.
Just answer in one word per sentence.

Examples:
sentence: "خیلی دیر به دستم رسید"
Sentiment: True

sentence: "خیلی خوش مزه بود"
Sentiment: False
...
"""

analyzer.update_templates(single_template=new_template)
print(analyzer.classify('راننده موتورش خیلی بد بود', method='single'))

```

### Error Handling
The implementation includes built-in retry mechanisms for rate limiting and API errors, with configurable maximum retry attempts.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.