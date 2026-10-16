import json
import logging
import os
from pathlib import Path
import random
import re
import time

from dotenv import load_dotenv
from httpx import HTTPStatusError, RequestError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

ROOT = Path(__file__).resolve().parent
LABELS = ("Positive", "Negative", "Neutral")
LOGGER = logging.getLogger(__name__)
SINGLE_TEMPLATE = """Classify this Persian review as Positive, Negative, or Neutral.
Examples: عالی بود -> Positive; خیلی دیر رسید -> Negative; معمولی بود -> Neutral.
Review: {sentence}
Return one label only."""
BATCH_TEMPLATE = """Classify each Persian review as Positive, Negative, or Neutral.
Examples: عالی بود -> Positive; خیلی دیر رسید -> Negative; معمولی بود -> Neutral.
Reviews (JSON array): {sentences}
Return a JSON array of labels, one per review, in the same order."""


class SentimentError(RuntimeError):
    pass


class InvalidResponseError(SentimentError):
    pass


class SentimentAnalyzer:
    def __init__(self, model_name=None, api_key=None, *, chat_model=None):
        if chat_model is None:
            load_dotenv(ROOT / ".env")
            api_key = (api_key or os.getenv("MISTRAL_API_KEY", "")).strip()
            if not api_key:
                raise ValueError("Set MISTRAL_API_KEY in .env or the environment.")
            model_name = model_name or os.getenv("MISTRAL_MODEL") or "mistral-large-latest"
            chat_model = ChatMistralAI(
                model=model_name, api_key=api_key,
                temperature=0, max_retries=0, timeout=60,
            )
        self.model = chat_model
        self.parser = StrOutputParser()
        self._single_template = SINGLE_TEMPLATE
        self._batch_template = BATCH_TEMPLATE
        self._single_labels = LABELS
        self._batch_labels = LABELS
        self.chain_single, self.chain_batch = self._build_chains(
            self._single_template, self._batch_template
        )

    def _build_chains(self, single_template, batch_template):
        single = ChatPromptTemplate.from_template(single_template)
        batch = ChatPromptTemplate.from_template(batch_template)
        if single.input_variables != ["sentence"]:
            raise ValueError("Single template must use only {sentence}.")
        if batch.input_variables != ["sentences"]:
            raise ValueError("Batch template must use only {sentences}.")
        return single | self.model | self.parser, batch | self.model | self.parser

    def update_templates(self, single_template=None, batch_template=None, *, labels=None):
        single = self._single_template if single_template is None else single_template
        batch = self._batch_template if batch_template is None else batch_template
        if labels is not None:
            if single_template is None and batch_template is None:
                raise ValueError("Supply a template when changing its labels.")
            if isinstance(labels, str):
                raise ValueError("Labels must be a sequence of distinct nonempty strings.")
            labels = tuple(labels)
            if not labels or any(not isinstance(label, str) or not label.strip() for label in labels):
                raise ValueError("Labels must be nonempty strings.")
            labels = tuple(label.strip() for label in labels)
            if len({label.casefold() for label in labels}) != len(labels):
                raise ValueError("Labels must be distinct.")
        chains = self._build_chains(single, batch)
        self._single_template, self._batch_template = single, batch
        self.chain_single, self.chain_batch = chains
        if labels is not None:
            if single_template is not None:
                self._single_labels = labels
            if batch_template is not None:
                self._batch_labels = labels

    def _invoke(self, chain, payload, max_retries):
        for attempt in range(max_retries + 1):
            try:
                return chain.invoke(payload).strip()
            except HTTPStatusError as error:
                status = error.response.status_code
                retryable = status == 429 or 500 <= status < 600
                message = f"Mistral request failed (HTTP {status})."
                if status in (401, 403):
                    message += " Check MISTRAL_API_KEY and account access."
                elif status == 404:
                    message += " Check MISTRAL_MODEL and model availability."
            except RequestError:
                retryable = True
                message = "Could not reach Mistral."
            if not retryable or attempt == max_retries:
                raise SentimentError(message) from None
            delay = min(2 ** attempt + random.random(), 30)
            LOGGER.warning("Mistral request failed; retrying in %.1f seconds.", delay)
            time.sleep(delay)

    @staticmethod
    def _label(value, labels):
        if isinstance(value, str):
            value = value.strip().strip("'\"").casefold()
            mapping = {label.casefold(): label for label in labels}
            if value in mapping:
                return mapping[value]
        raise InvalidResponseError("Model returned a label outside the configured categories.")

    def _batch_labels_from_response(self, response, count):
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if fenced:
            response = fenced.group(1)
        try:
            values = json.loads(response)
        except json.JSONDecodeError:
            values = [line.strip() for line in response.splitlines() if line.strip()]
        if not isinstance(values, list):
            raise InvalidResponseError("Batch response must be an array of labels.")
        if len(values) != count:
            raise InvalidResponseError(f"Expected {count} labels; received {len(values)}.")
        return [self._label(value, self._batch_labels) for value in values]

    def classify(self, comments, method="single", max_retries=5):
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        if method == "single":
            if not isinstance(comments, str) or not comments.strip():
                raise ValueError("Single input must be a nonempty string.")
            response = self._invoke(self.chain_single, {"sentence": comments}, max_retries)
            return self._label(response, self._single_labels)
        if method == "batch":
            if not isinstance(comments, (list, tuple)):
                raise ValueError("Batch input must be a list or tuple of strings.")
            if not comments:
                return []
            if any(not isinstance(comment, str) or not comment.strip() for comment in comments):
                raise ValueError("Each review must be a nonempty string.")
            response = self._invoke(
                self.chain_batch,
                {"sentences": json.dumps(list(comments), ensure_ascii=False)},
                max_retries,
            )
            return self._batch_labels_from_response(response, len(comments))
        raise ValueError("method must be 'single' or 'batch'.")


if __name__ == "__main__":
    from main import main

    main()
