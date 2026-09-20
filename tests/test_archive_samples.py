"""The archived text tasks ship synthetic demonstrations, never scraped text."""

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_archived_text_samples_are_explicitly_synthetic():
    rating = REPO / "archive/rating_estimation_from_text/data/review_synthetic.json"
    decision = REPO / "archive/review_decision_prediction/data/debate_synthetic.json"
    for path in (rating, decision):
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert rows
        assert all(row.get("source") == "synthetic_example" for row in rows)
    assert not (rating.parent / "review_amazon.json").exists()
    assert not (decision.parent / "debate.json").exists()


def test_archived_evaluators_default_to_synthetic_samples():
    rating_evaluator = (REPO / "archive/rating_estimation_from_text/reft_eva.py").read_text()
    decision_evaluator = (REPO / "archive/review_decision_prediction/rdp_eva.py").read_text()
    assert "data/review_synthetic.json" in rating_evaluator
    assert "data/debate_synthetic.json" in decision_evaluator
