import argparse
import json
from pathlib import Path

from reviews import classify_csv
from sentiment_analysis import SentimentAnalyzer, SentimentError


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Classify Persian reviews with the Mistral API.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", help="One review")
    inputs.add_argument("--file", type=Path, help="UTF-8 JSON array of reviews")
    inputs.add_argument("--csv", type=Path, help="CSV file of reviews")
    parser.add_argument("--output", type=Path, help="New CSV output file")
    parser.add_argument("--column", default="description", help="CSV text column")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--model", help="Override MISTRAL_MODEL")
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args(argv)
    if args.csv and args.output is None:
        parser.error("--csv requires --output")
    if args.max_retries < 0 or args.batch_size < 1:
        parser.error("--max-retries must be non-negative; --batch-size must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        comments = args.text
        if args.file:
            comments = json.loads(args.file.read_text(encoding="utf-8-sig"))
            if not isinstance(comments, list) or any(
                not isinstance(comment, str) or not comment.strip() for comment in comments
            ):
                raise ValueError("Input JSON must be an array of nonempty strings.")
        elif args.text is not None and not args.text.strip():
            raise ValueError("Review must not be empty.")
        analyzer = SentimentAnalyzer(model_name=args.model)
        if args.csv:
            count = classify_csv(
                analyzer, args.csv, args.output, text_column=args.column,
                batch_size=args.batch_size, max_retries=args.max_retries,
            )
            print(f"Saved {count} classified reviews to {args.output}")
        else:
            result = analyzer.classify(
                comments, method="batch" if args.file else "single",
                max_retries=args.max_retries,
            )
            print(json.dumps(result, ensure_ascii=False) if args.file else result)
    except (SentimentError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
