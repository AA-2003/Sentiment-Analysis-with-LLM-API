import asyncio
from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_mistralai import ChatMistralAI

import main
from sentiment_analysis import InvalidResponseError, SentimentAnalyzer, SentimentError


@contextmanager
def mistral_transport(handler):
    with httpx.Client(
        base_url="https://api.mistral.ai/v1",
        headers={"Authorization": "Bearer test-api-key"},
        transport=httpx.MockTransport(handler),
    ) as client:
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        model = ChatMistralAI(
            model="mistral-large-latest", api_key="test-api-key",
            client=client, async_client=async_client, max_retries=0, timeout=60,
        )
        try:
            yield SentimentAnalyzer(chat_model=model)
        finally:
            asyncio.run(async_client.aclose())


def completion(text):
    return httpx.Response(200, json={
        "id": "test-completion", "object": "chat.completion", "created": 0,
        "model": "mistral-large-latest",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


class SentimentTests(unittest.TestCase):
    def fake(self, *responses):
        return SentimentAnalyzer(chat_model=FakeListChatModel(responses=list(responses)))

    def test_actual_mistral_adapter_preserves_persian_text(self):
        requests = []
        def handler(request):
            requests.append(json.loads(request.content))
            return completion("Positive")
        with mistral_transport(handler) as analyzer:
            self.assertEqual(analyzer.classify("عالی بود"), "Positive")
        self.assertEqual(len(requests), 1)
        self.assertIn("عالی بود", json.dumps(requests[0], ensure_ascii=False))

    def test_batch_json_preserves_multiline_reviews_and_order(self):
        seen = []
        def respond(prompt):
            seen.append(prompt.to_string())
            return '["Negative", "Positive"]'
        analyzer = SentimentAnalyzer(chat_model=RunnableLambda(respond))
        comments = ['دیر رسید\nو سرد بود', 'گفت "عالی" و راضی بود']
        self.assertEqual(analyzer.classify(comments, method="batch"), ["Negative", "Positive"])
        self.assertIn(json.dumps(comments, ensure_ascii=False), seen[0])

    def test_legacy_newlines_and_fenced_json_are_supported(self):
        for response in ['Positive\n\nNegative', '```json\n["Positive","Negative"]\n```']:
            with self.subTest(response=response):
                self.assertEqual(
                    self.fake(response).classify(["خوب", "بد"], method="batch"),
                    ["Positive", "Negative"],
                )

    def test_invalid_batch_count_is_not_silently_saved_or_retried(self):
        respond = Mock(return_value='["Positive"]')
        analyzer = SentimentAnalyzer(chat_model=RunnableLambda(respond))
        with self.assertRaisesRegex(InvalidResponseError, "Expected 2 labels"):
            analyzer.classify(["خوب", "بد"], method="batch")
        self.assertEqual(respond.call_count, 1)

    def test_unknown_labels_and_non_array_json_are_rejected(self):
        for response in ['["Positive", "Unknown"]', '{"label": "Positive"}', '[true, false]']:
            with self.subTest(response=response):
                with self.assertRaises(InvalidResponseError):
                    self.fake(response).classify(["خوب", "بد"], method="batch")
        with self.assertRaises(InvalidResponseError):
            self.fake("The review seems positive.").classify("خوب بود")

    def test_labels_are_canonicalised(self):
        self.assertEqual(self.fake(' "positive" ').classify("خوب بود"), "Positive")

    def test_custom_single_labels_leave_batch_categories_intact(self):
        analyzer = self.fake("True", '["Positive", "Neutral"]')
        analyzer.update_templates(
            single_template="Is this about delivery? Return True or False: {sentence}",
            labels=("True", "False"),
        )
        self.assertEqual(analyzer.classify("دیر رسید"), "True")
        self.assertEqual(
            analyzer.classify(["خوب بود", "معمولی بود"], method="batch"),
            ["Positive", "Neutral"],
        )

    def test_invalid_template_update_keeps_existing_chain(self):
        analyzer = self.fake("Positive")
        with self.assertRaises(ValueError):
            analyzer.update_templates(single_template="Missing input variable")
        self.assertEqual(analyzer.classify("خوب بود"), "Positive")

    def test_invalid_inputs_make_no_model_calls(self):
        respond = Mock(return_value="Positive")
        analyzer = SentimentAnalyzer(chat_model=RunnableLambda(respond))
        for value, method in [([], "single"), ("", "single"), ("abc", "batch"),
                              (["x", None], "batch"), ([" "], "batch"), ("x", "unknown")]:
            with self.subTest(value=value, method=method):
                with self.assertRaises(ValueError):
                    analyzer.classify(value, method=method)
        self.assertEqual(analyzer.classify([], method="batch"), [])
        respond.assert_not_called()

    def test_missing_key_has_clear_error_before_client_creation(self):
        with patch("sentiment_analysis.load_dotenv"), patch.dict(os.environ, {}, clear=True):
            with patch("sentiment_analysis.ChatMistralAI") as client:
                with self.assertRaisesRegex(ValueError, "MISTRAL_API_KEY"):
                    SentimentAnalyzer()
                client.assert_not_called()

    def test_authentication_errors_fail_once_without_exposing_provider_body(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(401, json={"message": "sensitive-test-value"})
        with mistral_transport(handler) as analyzer, patch("sentiment_analysis.time.sleep") as sleep:
            with self.assertRaises(SentimentError) as error:
                analyzer.classify("خوب بود")
            self.assertIn("MISTRAL_API_KEY", str(error.exception))
            self.assertNotIn("sensitive-test-value", str(error.exception))
            sleep.assert_not_called()
        self.assertEqual(len(calls), 1)

    def test_retryable_errors_use_one_controlled_retry_budget(self):
        for failure in [429, 503, "network"]:
            with self.subTest(failure=failure):
                calls = []
                def handler(request):
                    calls.append(request)
                    if len(calls) == 1:
                        if failure == "network":
                            raise httpx.ConnectError("sensitive-test-value", request=request)
                        return httpx.Response(failure, json={"message": "temporary failure"})
                    return completion("Neutral")
                with mistral_transport(handler) as analyzer:
                    with patch("sentiment_analysis.time.sleep") as sleep:
                        self.assertEqual(analyzer.classify("معمولی بود", max_retries=1), "Neutral")
                        sleep.assert_called_once()
                self.assertEqual(len(calls), 2)

    def test_zero_retries_means_one_attempt_and_no_sleep(self):
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(429, json={"message": "rate limit"})
        with mistral_transport(handler) as analyzer, patch("sentiment_analysis.time.sleep") as sleep:
            with self.assertRaisesRegex(SentimentError, "HTTP 429"):
                analyzer.classify("خوب بود", max_retries=0)
            sleep.assert_not_called()
        self.assertEqual(len(calls), 1)

    def test_cli_json_uses_batch_mode_without_live_api(self):
        analyzer = self.fake('["Positive", "Neutral"]')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            path.write_text(json.dumps(["خوب بود", "معمولی بود"], ensure_ascii=False), encoding="utf-8")
            output = io.StringIO()
            with patch("main.SentimentAnalyzer", return_value=analyzer), redirect_stdout(output):
                main.main(["--file", str(path)])
            self.assertEqual(json.loads(output.getvalue()), ["Positive", "Neutral"])

    def test_cli_rejects_invalid_json_before_client_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.json"
            path.write_text('{"review": "x"}', encoding="utf-8")
            with patch("main.SentimentAnalyzer") as client:
                with self.assertRaises(SystemExit):
                    main.main(["--file", str(path)])
                client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
