"""Tests for newsclf.answer_key and scripts.evaluate_accuracy."""

from __future__ import annotations

import pandas as pd
import pytest

from newsclf.answer_key import (
    annotator_class1,
    canonicalise_label,
    compute_ranking,
    top1_from_ranking,
)
from scripts.evaluate_accuracy import _overlap_count

# ---------------------------------------------------------------------------
# Fixture: the example row from the spec
# ---------------------------------------------------------------------------

def _make_row(
    h1: list[str],
    h2: list[str],
    h3: list[str],
) -> pd.Series:
    """Build a pd.Series mimicking one row of classes.xlsx human columns."""
    data = {}
    for i, val in enumerate(h1):
        data[f"Human1Class{i + 1}"] = val
    for i in range(len(h1), 3):
        data[f"Human1Class{i + 1}"] = None
    for i, val in enumerate(h2):
        data[f"Hum2Class{i + 1}"] = val
    for i in range(len(h2), 3):
        data[f"Hum2Class{i + 1}"] = None
    for i, val in enumerate(h3):
        data[f"Hum3Class{i + 1}"] = val
    for i in range(len(h3), 3):
        data[f"Hum3Class{i + 1}"] = None
    return pd.Series(data)


@pytest.fixture()
def spec_row() -> pd.Series:
    """Row from the spec: H1=[Entertainment, Society and culture, Health],
    H2=[Health, Entertainment], H3=[Games, Entertainment, Health]."""
    return _make_row(
        h1=["Entertainment", "Society and culture", "Health"],
        h2=["Health", "Entertainment"],
        h3=["Games", "Entertainment", "Health"],
    )


# ---------------------------------------------------------------------------
# Tests: compute_ranking
# ---------------------------------------------------------------------------


class TestComputeRanking:
    def test_spec_example(self, spec_row: pd.Series) -> None:
        """Validate the spec example: Entertainment 7, Health 5, Games 3, Society and Culture 2."""
        ranking = compute_ranking(spec_row)
        assert ranking["Entertainment"] == 7
        assert ranking["Health"] == 5
        assert ranking["Games"] == 3
        assert ranking["Society and Culture"] == 2
        assert len(ranking) == 4

    def test_order_is_descending(self, spec_row: pd.Series) -> None:
        ranking = compute_ranking(spec_row)
        scores = list(ranking.values())
        assert scores == sorted(scores, reverse=True)

    def test_empty_positions_ignored(self) -> None:
        row = _make_row(h1=["Arts"], h2=[], h3=[])
        ranking = compute_ranking(row)
        assert ranking == {"Arts": 3}

    def test_all_empty(self) -> None:
        row = _make_row(h1=[], h2=[], h3=[])
        ranking = compute_ranking(row)
        assert ranking == {}


# ---------------------------------------------------------------------------
# Tests: capitalisation normalisation
# ---------------------------------------------------------------------------


class TestCanonicalisation:
    def test_lowercase_government(self) -> None:
        assert canonicalise_label("Government and politics") == "Government and Politics"

    def test_lowercase_society(self) -> None:
        assert canonicalise_label("Society and culture") == "Society and Culture"

    def test_already_canonical(self) -> None:
        assert canonicalise_label("Entertainment") == "Entertainment"

    def test_unknown_returns_none(self) -> None:
        assert canonicalise_label("Transportation") is None


# ---------------------------------------------------------------------------
# Tests: top1_from_ranking
# ---------------------------------------------------------------------------


class TestTop1:
    def test_no_tie(self, spec_row: pd.Series) -> None:
        ranking = compute_ranking(spec_row)
        top1, has_tie = top1_from_ranking(ranking)
        assert top1 == ["Entertainment"]
        assert has_tie is False

    def test_tie(self) -> None:
        ranking = {"Arts": 5, "Health": 5, "Sports": 3}
        top1, has_tie = top1_from_ranking(ranking)
        assert set(top1) == {"Arts", "Health"}
        assert has_tie is True

    def test_empty(self) -> None:
        top1, has_tie = top1_from_ranking({})
        assert top1 == []
        assert has_tie is False


# ---------------------------------------------------------------------------
# Tests: overlap count
# ---------------------------------------------------------------------------


class TestOverlap:
    def test_full_overlap(self) -> None:
        assert _overlap_count(["Arts", "Health", "Sports"], ["Arts", "Health", "Sports"]) == 3

    def test_partial_overlap(self) -> None:
        assert _overlap_count(["Arts", "Health", "Sports"], ["Arts", "Games"]) == 1

    def test_no_overlap(self) -> None:
        assert _overlap_count(["Arts", "Health", "Sports"], ["Games", "Travel"]) == 0

    def test_empty_pred(self) -> None:
        assert _overlap_count([], ["Arts", "Health"]) == 0


# ---------------------------------------------------------------------------
# Tests: leave-one-out (human baseline)
# ---------------------------------------------------------------------------


class TestLeaveOneOut:
    def test_two_agree_third_diverges(self) -> None:
        """H1 and H2 agree on Sports, H3 picks Arts.

        Leave-one-out for H3: answer key from H1+H2 = {Sports: 6}.
        H3 Class1 = Arts → miss.
        Leave-one-out for H1: answer key from H2+H3 = {Sports: 3, Arts: 3} → tie.
        H1 Class1 = Sports → hit (Sports is in the tie set).
        """
        row = _make_row(
            h1=["Sports"],
            h2=["Sports"],
            h3=["Arts"],
        )
        # H3 diverges: leave-one-out answer key = H1+H2 = Sports:6
        ranking_without_h3 = compute_ranking(row, annotators=["Human1", "Hum2"])
        top1, _ = top1_from_ranking(ranking_without_h3)
        assert top1 == ["Sports"]
        h3_pick = annotator_class1(row, "Hum3")
        assert h3_pick == "Arts"
        assert h3_pick not in top1  # H3 misses

        # H1 agrees: leave-one-out answer key = H2+H3 = {Sports:3, Arts:3}
        ranking_without_h1 = compute_ranking(row, annotators=["Hum2", "Hum3"])
        top1_h1, has_tie = top1_from_ranking(ranking_without_h1)
        assert set(top1_h1) == {"Sports", "Arts"}
        assert has_tie is True
        h1_pick = annotator_class1(row, "Human1")
        assert h1_pick == "Sports"
        assert h1_pick in top1_h1  # H1 hits (Sports in tie)

    def test_leave_one_out_tie_hit(self) -> None:
        """All three annotators pick the same Class1 → everyone hits."""
        row = _make_row(h1=["Health"], h2=["Health"], h3=["Health"])
        for ann, others in [
            ("Human1", ["Hum2", "Hum3"]),
            ("Hum2", ["Human1", "Hum3"]),
            ("Hum3", ["Human1", "Hum2"]),
        ]:
            ranking = compute_ranking(row, annotators=others)
            top1, _ = top1_from_ranking(ranking)
            pick = annotator_class1(row, ann)
            assert pick in top1

    def test_annotator_class1_empty(self) -> None:
        """Annotator with no Class1 returns None."""
        row = _make_row(h1=[], h2=["Arts"], h3=["Arts"])
        assert annotator_class1(row, "Human1") is None

    def test_subset_ranking(self) -> None:
        """compute_ranking with a subset only includes those annotators."""
        row = _make_row(
            h1=["Entertainment", "Health"],
            h2=["Sports"],
            h3=["Games"],
        )
        ranking_h1_only = compute_ranking(row, annotators=["Human1"])
        assert ranking_h1_only == {"Entertainment": 3, "Health": 2}
        assert "Sports" not in ranking_h1_only
        assert "Games" not in ranking_h1_only
