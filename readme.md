# Sentiment Analysis with LLM API

This repository contains a simple implementation of sentiment analysis using a Large Language Model (LLM) API. It labels comments based on their sentiment by leveraging the [Mistral AI](https://console.mistral.ai) API. The implementation is built using [LangChain](https://python.langchain.com/) and supports two methods of API calls:

1. **Single API Call per Comment** - Each comment is sent individually to the API for analysis.
2. **Batch API Calls** - Comments are grouped and processed in a single API request for efficiency.

## Setup

### Prerequisites
- Python 3.8+
- [LangChain](https://python.langchain.com/)
- An API key from [Mistral AI](https://console.mistral.ai) (sign up for their free plan)

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
   - Create a `.env` file in the root directory.
   - Add the following line to the `.env` file:
     ```sh
     MISTRAL_API_KEY=your_api_key_here
     ```

## Usage

Run the script to analyze comments:
```sh
python sentiment_analysis.py
```
or you can use the `sentiment_analysis.py` file as a module in your own project.

The script will process comments using one of the two available methods (single or batch API calls) and output labeled sentiment results.



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.