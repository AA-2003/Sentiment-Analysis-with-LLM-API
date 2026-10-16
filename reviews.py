from pathlib import Path
import re

import emoji
import pandas as pd


def preprocess_text(text):
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]|\d+", " ", text)
    return " ".join(text.split())


def prepare_reviews(frame, max_per_rating=5000, random_state=42):
    if isinstance(max_per_rating, bool) or not isinstance(max_per_rating, int) or max_per_rating < 1:
        raise ValueError("max_per_rating must be a positive integer.")
    missing = {"description", "star"} - set(frame.columns)
    if missing:
        raise ValueError("Missing review columns: " + ", ".join(sorted(missing)))
    frame = frame.dropna(subset=["description", "star"]).copy()
    mask = frame["description"].map(lambda value: isinstance(value, str)).astype(bool)
    frame = frame.loc[mask].copy()
    if frame.empty:
        raise ValueError("No usable reviews remain after preprocessing.")
    frame = frame.drop_duplicates("description")
    frame["preprocessed_description"] = frame["description"].map(preprocess_text)
    frame = frame[frame["preprocessed_description"].str.len() > 0]
    frame = frame.drop_duplicates("preprocessed_description")
    if frame.empty:
        raise ValueError("No usable reviews remain after preprocessing.")
    count = min(int(frame["star"].value_counts().min()), max_per_rating)
    groups = [
        group.sample(n=count, random_state=random_state)
        for _, group in frame.groupby("star", sort=True)
    ]
    return pd.concat(groups, ignore_index=True)


def classify_csv(analyzer, input_path, output_path, *, text_column="description",
                 batch_size=5, chunk_size=100, max_retries=5):
    for name, value in [("batch_size", batch_size), ("chunk_size", chunk_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different.")
    columns = pd.read_csv(input_path, nrows=0).columns
    if text_column not in columns:
        raise ValueError(f"Missing text column: {text_column}")
    if "sentiment" in columns:
        raise ValueError("Input already contains a sentiment column.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        output_path.open("x", encoding="utf-8", newline="") as handle,
        pd.read_csv(input_path, chunksize=chunk_size) as chunks,
    ):
        pd.DataFrame(columns=[*columns, "sentiment"]).to_csv(handle, index=False)
        for chunk in chunks:
            for start in range(0, len(chunk), batch_size):
                batch = chunk.iloc[start:start + batch_size].copy()
                results = analyzer.classify(
                    batch[text_column].tolist(), method="batch", max_retries=max_retries
                )
                if not isinstance(results, list) or len(results) != len(batch):
                    raise ValueError("Result count does not match the CSV batch.")
                batch["sentiment"] = results
                batch.to_csv(handle, index=False, header=False)
                handle.flush()
                count += len(batch)
    return count
