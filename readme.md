# Persian Sentiment Analysis

Classify Persian reviews as Positive, Negative, or Neutral using
[Mistral](https://docs.mistral.ai/) and LangChain.

## Setup

Use Python 3.11 or 3.12.

```bash
git clone https://github.com/AA-2003/Sentiment-Analysis-with-LLM-API.git
cd Sentiment-Analysis-with-LLM-API
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on macOS / Linux, then install:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `MISTRAL_API_KEY`.
The default model is `mistral-large-latest`; change `MISTRAL_MODEL` or use `--model`.

## Usage

```bash
python main.py --text "عالی بود"
python main.py --file reviews.json
python main.py --csv data/reviews.csv --column description --output reports/results.csv
```

The JSON input is an array of review strings. CSV processing makes one request
per batch (default: 5 reviews). Output files must be new; completed batches
remain available if a later request fails.

```python
from sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.classify("عالی بود")
analyzer.classify(["عالی بود", "دیر رسید", "معمولی بود"], method="batch")
```

Responses are checked for valid labels and the expected number of results.
HTTP 429, server errors, and connection errors use exponential backoff.
`max_retries=0` makes one attempt; the default permits five retries.
Authentication and invalid-request errors are not retried.

Custom templates use `{sentence}` for single inputs or `{sentences}` for batches.
Pass `labels` when changing categories:

```python
analyzer.update_templates(
    single_template='Return True if this review concerns delivery, otherwise False: {sentence}',
    labels=("True", "False"),
)
```

## Data and notebooks

`reviews.py` cleans emoji, URLs, punctuation, and digits, removes duplicate
reviews, and balances star-rating groups with a fixed random seed.
It preserves the CSV preparation workflow from `ex.ipynb`.

```bash
python -m pip install -r requirements-dev.txt
jupyter lab
python -m unittest discover -s tests -v
```

`main.ipynb` demonstrates single, batch, and custom-label calls.
`ex.ipynb` demonstrates preprocessing and CSV classification.
Both use fixed mock responses by default; set `LIVE = True` to call Mistral.
Live requests use your account's API quota. Tests and CI make no model API calls.
