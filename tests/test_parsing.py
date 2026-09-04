"""Extraction of the Final Judgment from replies in the many shapes models produce."""

import pytest

from socialmaze.hrd.parsing import (
    Prediction,
    extract_criminal,
    extract_role,
    normalize_role,
    parse_final_judgment,
    roles_in_phrase,
)
from socialmaze.hrd.prompts import FINAL_JUDGMENT_FORMAT, answer_block
from socialmaze.hrd.rules import CRIMINAL, INVESTIGATOR, LUNATIC, RUMORMONGER, UNKNOWN


@pytest.mark.parametrize("word,role", [
    ("Investigator", INVESTIGATOR), ("investigators", INVESTIGATOR), ("INVESTIGATOR", INVESTIGATOR),
    ("Criminal", CRIMINAL), ("killer", CRIMINAL), ("Murderer", CRIMINAL),
    ("Rumormonger", RUMORMONGER), ("rumour-monger", RUMORMONGER),
    ("Lunatic", LUNATIC), ("lunatics", LUNATIC),
    ("Unknown", UNKNOWN), ("uncertain", UNKNOWN), ("Unsure", UNKNOWN), ("undetermined", UNKNOWN),
    ("Wizard", None), ("the", None), ("an", None), ("", None),
])
def test_normalize_role(word, role):
    assert normalize_role(word) == role


@pytest.mark.parametrize("text,criminal,role", [
    ("Final Judgment:\nFinal Criminal Is Player 4.\nMy Role Is Lunatic.", 4, LUNATIC),
    (answer_block(2, RUMORMONGER), 2, RUMORMONGER),
    ("**Final Criminal Is Player 4.**\n**My Role Is Investigator.**", 4, INVESTIGATOR),
    ("Final Criminal Is Player #4\nMy Role Is Rumormonger", 4, RUMORMONGER),
    ("final criminal is player 4.\nmy role is rumormonger.", 4, RUMORMONGER),
    ("Final Criminal Is: Player 4\nMy Role Is: Lunatic", 4, LUNATIC),
    ("### Final Judgment\nFinal Criminal Is **Player 6**.\nMy Role Is **Lunatic**", 6, LUNATIC),
    ("Final Criminal Is Player 10.\nMy Role Is the Criminal.", 10, CRIMINAL),
    ("Final Criminal Is Player 4.\nMy Role Is an Investigator.", 4, INVESTIGATOR),
    ("Final Criminal Is Player 4.\nMy Role Is [Lunatic].", 4, LUNATIC),
    ("Final Criminal Is Player 4.\nMy Role Is \"Lunatic\".", 4, LUNATIC),
    ("Final Criminal Is Player 4.\nMy Role Is Lunatic, not the Criminal.", 4, LUNATIC),
    ("Final Criminal Is Player 4.\nMy Role Is Lunatic (I was told Criminal).", 4, LUNATIC),
    ("Final Criminal Is Player 4. My Role Is Lunatic. Thank you.", 4, LUNATIC),
    ("Final Criminal Is Player 4.\nMy Role Is Unknown.", 4, UNKNOWN),
    ("Final Criminal Is Player 4.\nMy Role Is uncertain.", 4, UNKNOWN),
])
def test_parse_final_judgment_formats(text, criminal, role):
    pred = parse_final_judgment(text)
    assert pred == Prediction(criminal=criminal, role=role, found=True, hedged=False)


def test_last_occurrence_wins():
    text = (
        "At first I thought:\nFinal Criminal Is Player 2.\nMy Role Is Investigator.\n"
        "But the Investigator count rules that out.\n\n"
        "Final Judgment:\nFinal Criminal Is Player 5.\nMy Role Is Rumormonger."
    )
    assert parse_final_judgment(text) == Prediction(5, RUMORMONGER, True, False)


@pytest.mark.parametrize("line", [
    "My Role Is Investigator or Rumormonger.",
    "My Role Is either an Investigator or a Rumormonger",
    "My Role Is Investigator/Rumormonger",
    "My Role Is **Criminal or Lunatic**",
    "My Role Is [Investigator/Criminal/Rumormonger/Lunatic/Unknown].",
    "My Role Is Unknown or Rumormonger",
])
def test_hedged_role_lines(line):
    pred = parse_final_judgment(f"Final Criminal Is Player 3.\n{line}")
    assert pred.found and pred.criminal == 3
    assert pred.hedged is True
    assert pred.role is None


def test_unrecognised_role_is_none_not_hedged():
    pred = parse_final_judgment("Final Criminal Is Player 3.\nMy Role Is Wizard.")
    assert pred == Prediction(3, None, True, False)


def test_missing_block():
    assert parse_final_judgment("I need more information.") == Prediction(None, None, False, False)
    assert parse_final_judgment("") == Prediction(None, None, False, False)
    assert parse_final_judgment(None) == Prediction(None, None, False, False)


def test_role_without_criminal_line_is_not_found():
    pred = parse_final_judgment("My Role Is Lunatic.")
    assert pred.found is False and pred.criminal is None and pred.role == LUNATIC


def test_template_placeholders_do_not_match():
    pred = parse_final_judgment(FINAL_JUDGMENT_FORMAT)
    assert pred.found is False
    assert pred.hedged is True


def test_criminal_line_split_across_markdown_and_newline():
    assert extract_criminal("Final Criminal Is\n**Player 7**") == 7
    assert extract_criminal("Final Criminal Is Player 12, My Role Is Lunatic") == 12


def test_wrappers():
    text = "Final Criminal Is Player 4.\nMy Role Is Lunatic."
    assert extract_criminal(text) == 4
    assert extract_role(text) == LUNATIC
    assert extract_criminal("nothing") is None
    assert extract_role("Final Criminal Is Player 4.") is None


def test_roles_in_phrase():
    assert roles_in_phrase(" the Criminal.") == [CRIMINAL]
    assert roles_in_phrase("Investigator or Rumormonger") == [INVESTIGATOR, RUMORMONGER]
    assert roles_in_phrase("Investigator, or maybe Rumormonger") == [INVESTIGATOR]
    assert roles_in_phrase("Investigator or investigator") == [INVESTIGATOR]
    assert roles_in_phrase("") == []
