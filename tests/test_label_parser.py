"""Label normalization / parser tests."""

from room_vlm.constants import CANONICAL_LABELS
from room_vlm.data.collator import normalize_predicted_label


def test_valid_label():
    assert normalize_predicted_label("bedroom", CANONICAL_LABELS, strict=True) == "bedroom"
    assert normalize_predicted_label("Living Room", CANONICAL_LABELS, strict=True) == "living room"


def test_invalid_strict():
    assert (
        normalize_predicted_label("This looks like a bedroom.", CANONICAL_LABELS, strict=True)
        == "invalid"
    )
