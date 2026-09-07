"""
Tests for interface/command_parser.py.

Placeholder for future tests covering parsing of user commands into
structured object/attribute search requests.

Not yet implemented.
"""

"""Tests for interface/command_parser.py."""

from interface.command_parser import parse_command


def test_parse_class_without_color():
    result = parse_command("find cat")

    assert result["target_class"] == "cat"
    assert result["target_color"] is None


def test_parse_class_with_filler_words():
    result = parse_command("find my cat")

    assert result["target_class"] == "cat"
    assert result["target_color"] is None


def test_parse_class_and_color():
    result = parse_command("find the orange cat")

    assert result["target_class"] == "cat"
    assert result["target_color"] == "orange"


def test_parse_track_command():
    result = parse_command("track the red bottle")

    assert result["target_class"] == "bottle"
    assert result["target_color"] == "red"


def test_parse_person_with_color():
    result = parse_command("find a person")

    assert result["target_class"] == "person"
    assert result["target_color"] is None


def test_parse_person_wearing_color():
    result = parse_command("track the person wearing blue")

    assert result["target_class"] == "person"
    assert result["target_color"] == "blue"


def test_parse_is_case_insensitive():
    result = parse_command("FIND THE ORANGE CAT")

    assert result["target_class"] == "cat"
    assert result["target_color"] == "orange"


def test_parse_handles_extra_whitespace():
    result = parse_command("   find   cat   ")

    assert result["target_class"] == "cat"
    assert result["target_color"] is None


def test_missing_color_returns_none():
    result = parse_command("track the bottle")

    assert result["target_class"] == "bottle"
    assert result["target_color"] is None


def test_unknown_class_returns_none():
    result = parse_command("find spaceship")

    assert result["target_class"] is None
    assert result["target_color"] is None