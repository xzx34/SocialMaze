from socialmaze.hrd.prompts import (
    FINAL_JUDGMENT_FORMAT,
    answer_block,
    final_message,
    role_count_phrase,
    round_block,
    round_message,
    system_prompt,
)
from socialmaze.hrd.rules import CRIMINAL, INVESTIGATOR, GameConfig, Statement

STS = [Statement(1, 2, "is"), Statement(2, 3, "is_not"), Statement(3, 1, "is_not")]


def test_role_count_phrase_pluralises_and_keeps_zero_counts():
    assert role_count_phrase(GameConfig.create(6, "full")) == "3 Investigators, 1 Criminal, 1 Rumormonger, 1 Lunatic"
    assert role_count_phrase(GameConfig.create(6, "original")) == "5 Investigators, 1 Criminal, 0 Rumormongers, 0 Lunatics"
    assert role_count_phrase(GameConfig.create(10, "full")) == "5 Investigators, 1 Criminal, 2 Rumormongers, 2 Lunatics"


def test_system_prompt_mentions_setup_role_and_format():
    text = system_prompt(GameConfig.create(6, "full"), CRIMINAL)
    assert "6 players in total" in text
    assert "3 Investigators, 1 Criminal, 1 Rumormonger, 1 Lunatic" in text
    assert "you have been told that you are the Criminal" in text
    assert "Only Investigators are guaranteed to be truthful" in text
    assert text.endswith(FINAL_JUDGMENT_FORMAT)
    text_i = system_prompt(GameConfig.create(6, "full"), INVESTIGATOR)
    assert "you have been told that you are an Investigator" in text_i


def test_round_and_final_messages():
    assert round_block(2, STS) == (
        "Round 2 statements:\n"
        "Player 1 says Player 2 is the criminal.\n"
        "Player 2 says Player 3 is not the criminal.\n"
        "Player 3 says Player 1 is not the criminal."
    )
    assert "(round 1)" in round_message(1, STS)
    assert "(rounds 1 to 3)" in round_message(3, STS)
    fm = final_message([STS, STS])
    assert fm.startswith("Round 1 statements:\n") and "\n\nRound 2 statements:\n" in fm
    assert "all 2 rounds" in fm
    for msg in (round_message(1, STS), fm):
        assert msg.rstrip().endswith("your Final Judgment.")


def test_answer_block():
    assert answer_block(4, "Lunatic") == "Final Judgment:\nFinal Criminal Is Player 4.\nMy Role Is Lunatic."
    assert answer_block(4, "Lunatic", header=False) == "Final Criminal Is Player 4.\nMy Role Is Lunatic."
