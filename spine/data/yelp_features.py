"""Feature extraction for Yelp subset."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Iterator

import json

from pydantic import BaseModel, Field, ValidationError


class UserStats(BaseModel):
    user_id: str
    review_count: int = Field(ge=0)
    average_stars: float = Field(ge=1.0, le=5.0)


class BusinessStats(BaseModel):
    business_id: str
    stars: float = Field(ge=1.0, le=5.0)
    review_count: int = Field(ge=0)
    is_open: int = Field(ge=0, le=1)
    categories_count: int = Field(ge=0)


@dataclass(frozen=True)
class FeatureRow:
    review_id: str
    user_id: str
    business_id: str
    review_date: datetime
    review_stars: float
    label: int
    review_len_words: int
    review_len_chars: int
    review_useful: int
    review_funny: int
    review_cool: int
    user_review_count: int
    user_average_stars: float
    business_review_count: int
    business_stars: float
    business_is_open: int
    business_categories_count: int

    def to_feature_vector(self) -> list[float]:
        return [
            float(self.review_len_words),
            float(self.review_len_chars),
            float(self.review_useful),
            float(self.review_funny),
            float(self.review_cool),
            float(self.user_review_count),
            float(self.user_average_stars),
            float(self.business_review_count),
            float(self.business_stars),
            float(self.business_is_open),
            float(self.business_categories_count),
        ]


FEATURE_COLUMNS = [
    "review_len_words",
    "review_len_chars",
    "review_useful",
    "review_funny",
    "review_cool",
    "user_review_count",
    "user_average_stars",
    "business_review_count",
    "business_stars",
    "business_is_open",
    "business_categories_count",
]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_user_stats(path: Path) -> dict[str, UserStats]:
    stats: dict[str, UserStats] = {}
    for row in iter_jsonl(path):
        avg = float(row.get("average_stars", 3.0) or 3.0)
        avg = min(max(avg, 1.0), 5.0)
        try:
            parsed = UserStats(
                user_id=row["user_id"],
                review_count=row.get("review_count", 0),
                average_stars=avg,
            )
        except (ValidationError, KeyError):
            continue
        stats[parsed.user_id] = parsed
    return stats


def load_business_stats(path: Path) -> dict[str, BusinessStats]:
    stats: dict[str, BusinessStats] = {}
    for row in iter_jsonl(path):
        categories = row.get("categories") or ""
        categories_count = len([c for c in categories.split(",") if c.strip()])
        stars = float(row.get("stars", 3.0) or 3.0)
        stars = min(max(stars, 1.0), 5.0)
        try:
            parsed = BusinessStats(
                business_id=row["business_id"],
                stars=stars,
                review_count=row.get("review_count", 0),
                is_open=int(row.get("is_open", 0) or 0),
                categories_count=categories_count,
            )
        except (ValidationError, KeyError):
            continue
        stats[parsed.business_id] = parsed
    return stats


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


# region book:fnd.ml-data-and-evaluation.example-01
from pathlib import Path


def build_feature_rows(
    subset_dir: Path,
    label_threshold: float = 4.0,
    max_rows: int | None = None,
) -> list[FeatureRow]:
    review_path = subset_dir / "review.json"
    business_path = subset_dir / "business.json"
    user_path = subset_dir / "user.json"

    user_stats = load_user_stats(user_path)
    business_stats = load_business_stats(business_path)

    rows: list[FeatureRow] = []
    for idx, row in enumerate(iter_jsonl(review_path)):
        if max_rows is not None and idx >= max_rows:
            break
        text = row.get("text") or ""
        stars = float(row.get("stars", 0.0))
        label = 1 if stars >= label_threshold else 0

        user = user_stats.get(
            row.get("user_id", ""),
            UserStats(user_id="", review_count=0, average_stars=3.0),
        )
        business = business_stats.get(
            row.get("business_id", ""),
            BusinessStats(
                business_id="",
                stars=3.0,
                review_count=0,
                is_open=0,
                categories_count=0,
            ),
        )

        rows.append(
            FeatureRow(
                review_id=row.get("review_id", ""),
                user_id=row.get("user_id", ""),
                business_id=row.get("business_id", ""),
                review_date=_parse_date(row.get("date", "1970-01-01 00:00:00")),
                review_stars=stars,
                label=label,
                review_len_words=len(text.split()),
                review_len_chars=len(text),
                review_useful=int(row.get("useful", 0) or 0),
                review_funny=int(row.get("funny", 0) or 0),
                review_cool=int(row.get("cool", 0) or 0),
                user_review_count=user.review_count,
                user_average_stars=user.average_stars,
                business_review_count=business.review_count,
                business_stars=business.stars,
                business_is_open=business.is_open,
                business_categories_count=business.categories_count,
            )
        )
    return rows
# endregion


# region book:fnd.ml-data-and-evaluation.core-01
def time_based_split(
    rows: list[FeatureRow], test_fraction: float = 0.2
) -> tuple[list[FeatureRow], list[FeatureRow]]:
    if not rows:
        return [], []
    rows_sorted = sorted(rows, key=lambda row: row.review_date)
    split_idx = max(1, int(len(rows_sorted) * (1.0 - test_fraction)))
    return rows_sorted[:split_idx], rows_sorted[split_idx:]
# endregion
