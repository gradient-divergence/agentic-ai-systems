"""Build a deterministic Yelp Open Dataset subset for the spine."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, Field, ValidationError, field_validator


class SubsetConfig(BaseModel):
    input_dir: Path = Field(..., description="Directory with Yelp JSONL files.")
    output_dir: Path = Field(..., description="Output directory for the subset.")
    reviews_target: int = Field(100_000, ge=1)
    seed: int = Field(13, ge=0)
    min_stars: float | None = Field(None, ge=1.0, le=5.0)
    max_stars: float | None = Field(None, ge=1.0, le=5.0)
    review_file: str = "review.json"
    business_file: str = "business.json"
    user_file: str = "user.json"

    @field_validator("input_dir")
    @classmethod
    def _validate_input_dir(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"input_dir does not exist: {value}")
        return value

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value

    @property
    def review_path(self) -> Path:
        return self.input_dir / self.review_file

    @property
    def business_path(self) -> Path:
        return self.input_dir / self.business_file

    @property
    def user_path(self) -> Path:
        return self.input_dir / self.user_file


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def filter_reviews(
    rows: Iterable[dict[str, Any]],
    min_stars: float | None,
    max_stars: float | None,
) -> Iterator[dict[str, Any]]:
    for row in rows:
        stars = row.get("stars")
        if stars is None:
            continue
        if min_stars is not None and stars < min_stars:
            continue
        if max_stars is not None and stars > max_stars:
            continue
        yield row


def reservoir_sample(
    rows: Iterable[dict[str, Any]], target: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if idx < target:
            sample.append(row)
        else:
            j = rng.randint(0, idx)
            if j < target:
                sample[j] = row
    return sample


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def _resolve_input_file(input_dir: Path, preferred: str, fallback: str) -> Path:
    preferred_path = input_dir / preferred
    if preferred_path.exists():
        return preferred_path
    fallback_path = input_dir / fallback
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"Missing input file. Tried {preferred_path.name} and {fallback_path.name}."
    )


def build_subset(cfg: SubsetConfig) -> None:
    review_path = _resolve_input_file(
        cfg.input_dir,
        cfg.review_file,
        "yelp_academic_dataset_review.json",
    )
    business_path = _resolve_input_file(
        cfg.input_dir,
        cfg.business_file,
        "yelp_academic_dataset_business.json",
    )
    user_path = _resolve_input_file(
        cfg.input_dir,
        cfg.user_file,
        "yelp_academic_dataset_user.json",
    )

    reviews_iter = iter_jsonl(review_path)
    filtered = filter_reviews(reviews_iter, cfg.min_stars, cfg.max_stars)
    sampled_reviews = reservoir_sample(filtered, cfg.reviews_target, cfg.seed)

    business_ids = {row["business_id"] for row in sampled_reviews if "business_id" in row}
    user_ids = {row["user_id"] for row in sampled_reviews if "user_id" in row}

    businesses = (row for row in iter_jsonl(business_path) if row.get("business_id") in business_ids)
    users = (row for row in iter_jsonl(user_path) if row.get("user_id") in user_ids)

    reviews_out = cfg.output_dir / "review.json"
    business_out = cfg.output_dir / "business.json"
    user_out = cfg.output_dir / "user.json"

    review_count = write_jsonl(reviews_out, sampled_reviews)
    business_count = write_jsonl(business_out, businesses)
    user_count = write_jsonl(user_out, users)

    metadata = {
        "dataset": "yelp-dataset",
        "reviews_target": cfg.reviews_target,
        "reviews_selected": review_count,
        "businesses_selected": business_count,
        "users_selected": user_count,
        "seed": cfg.seed,
        "min_stars": cfg.min_stars,
        "max_stars": cfg.max_stars,
        "review_file": review_path.name,
        "business_file": business_path.name,
        "user_file": user_path.name,
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
    }
    with (cfg.output_dir / "subset_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args() -> SubsetConfig:
    parser = argparse.ArgumentParser(description="Build Yelp Open Dataset subset.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reviews-target", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-stars", type=float, default=None)
    parser.add_argument("--max-stars", type=float, default=None)
    parser.add_argument("--review-file", default="review.json")
    parser.add_argument("--business-file", default="business.json")
    parser.add_argument("--user-file", default="user.json")
    args = parser.parse_args()
    try:
        return SubsetConfig(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            reviews_target=args.reviews_target,
            seed=args.seed,
            min_stars=args.min_stars,
            max_stars=args.max_stars,
            review_file=args.review_file,
            business_file=args.business_file,
            user_file=args.user_file,
        )
    except ValidationError as exc:
        raise SystemExit(str(exc)) from exc


def main() -> None:
    cfg = parse_args()
    build_subset(cfg)


if __name__ == "__main__":
    main()
