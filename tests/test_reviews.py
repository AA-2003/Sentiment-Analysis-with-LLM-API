from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd

from reviews import classify_csv, prepare_reviews, preprocess_text
from sentiment_analysis import SentimentError


class ReviewTests(unittest.TestCase):
    def test_preprocessing_preserves_persian_words_and_negation(self):
        self.assertEqual(
            preprocess_text("خیلی 😊 خوب نبود! https://example.invalid/12 ۱۲۳"),
            "خیلی خوب نبود",
        )

    def test_preparation_deduplicates_balances_and_preserves_source_ids(self):
        frame = pd.DataFrame({
            "_id": list(range(8)),
            "description": ["عالی 😊", "عالی!!!", "خوب بود", "بد بود",
                            "دیر رسید", "معمولی بود", None, "😀"],
            "star": [5, 5, 5, 1, 1, 3, 3, 1],
        })
        result = prepare_reviews(frame)
        self.assertEqual(result["star"].value_counts().to_dict(), {1: 1, 3: 1, 5: 1})
        self.assertEqual(result["preprocessed_description"].nunique(), 3)
        self.assertTrue(set(result["_id"]).issubset(set(frame["_id"])))
        pd.testing.assert_frame_equal(result, prepare_reviews(frame))

    def test_empty_and_unusable_review_sets_raise_clear_errors(self):
        cases = [
            pd.DataFrame(columns=["description", "star"]),
            pd.DataFrame({"description": [None, 42, "😀"], "star": [1, 3, 5]}),
            pd.DataFrame({"description": ["text"]}),
        ]
        for frame in cases:
            with self.subTest(rows=len(frame)):
                with self.assertRaises(ValueError):
                    prepare_reviews(frame)

    def test_csv_keeps_row_order_and_labels_across_chunk_boundaries(self):
        frame = pd.DataFrame({"description": ["a", "b", "c", "d", "e"], "id": range(5)})
        analyzer = Mock()
        analyzer.classify.side_effect = [
            ["Positive", "Negative"], ["Neutral"], ["Negative", "Positive"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.csv", Path(directory) / "output.csv"
            frame.to_csv(source, index=False)
            count = classify_csv(analyzer, source, output, batch_size=2, chunk_size=3)
            result = pd.read_csv(output)
            self.assertEqual(count, 5)
            self.assertEqual(result["id"].tolist(), list(range(5)))
            self.assertEqual(
                result["sentiment"].tolist(),
                ["Positive", "Negative", "Neutral", "Negative", "Positive"],
            )

    def test_csv_refuses_existing_output_without_appending_or_calling_model(self):
        analyzer = Mock()
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.csv", Path(directory) / "output.csv"
            pd.DataFrame({"description": ["text"]}).to_csv(source, index=False)
            output.write_text("existing results\n", encoding="utf-8")
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                classify_csv(analyzer, source, output)
            self.assertEqual(output.read_bytes(), original)
            analyzer.classify.assert_not_called()

    def test_completed_batches_are_preserved_after_api_failure(self):
        analyzer = Mock()
        analyzer.classify.side_effect = [
            ["Positive", "Negative"], SentimentError("HTTP 429"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.csv", Path(directory) / "output.csv"
            pd.DataFrame({"description": ["a", "b", "c"]}).to_csv(source, index=False)
            with self.assertRaises(SentimentError):
                classify_csv(analyzer, source, output, batch_size=2)
            result = pd.read_csv(output)
            self.assertEqual(result["description"].tolist(), ["a", "b"])
            self.assertEqual(len(result), 2)

    def test_mismatched_results_do_not_write_misaligned_rows(self):
        analyzer = Mock()
        analyzer.classify.return_value = ["Positive"]
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.csv", Path(directory) / "output.csv"
            pd.DataFrame({"description": ["a", "b"]}).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "Result count"):
                classify_csv(analyzer, source, output)
            self.assertEqual(len(pd.read_csv(output)), 0)


if __name__ == "__main__":
    unittest.main()
