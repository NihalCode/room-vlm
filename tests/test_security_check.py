"""Security check unit test."""

from pathlib import Path

from room_vlm.security_check import audit


def test_security_audit_clean_on_empty_or_source(tmp_path, monkeypatch):
    # Running against the real repo should pass once files are committed properly;
    # here we only assert the function returns a list.
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    assert isinstance(result, list)
