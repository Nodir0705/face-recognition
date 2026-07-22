"""Tests for the URL parser in src.sheets — the only piece that doesn't need
the Google API client to be installed."""

import pytest
from src.sheets import parse_sheet_input


def test_parse_full_edit_url():
    url = "https://docs.google.com/spreadsheets/d/1aBcDeFgHi-Jk_lMnoPqRsTuVwXyZ0123456789/edit#gid=0"
    assert parse_sheet_input(url) == "1aBcDeFgHi-Jk_lMnoPqRsTuVwXyZ0123456789"


def test_parse_url_with_extra_path():
    url = "https://docs.google.com/spreadsheets/d/abc-DEF_123456789012345678/edit?usp=sharing"
    assert parse_sheet_input(url) == "abc-DEF_123456789012345678"


def test_parse_bare_id():
    sid = "1aBcDeFgHi-Jk_lMnoPqRsTuVwXyZ0123456789"
    assert parse_sheet_input(sid) == sid


def test_parse_strips_whitespace():
    url = "  https://docs.google.com/spreadsheets/d/abc12345678901234567890/  "
    assert parse_sheet_input(url) == "abc12345678901234567890"


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_sheet_input("")
    with pytest.raises(ValueError):
        parse_sheet_input("not-a-url-or-id")
    with pytest.raises(ValueError):
        parse_sheet_input("https://example.com/")
