"""Loss masking unit test (no model download)."""

from room_vlm.constants import IGNORE_INDEX
from room_vlm.data.collator import build_masked_labels


def test_prompt_tokens_masked_answer_trainable():
    # Fake token ids: prompt [1,2,3,4] then answer [10,11]
    full = [1, 2, 3, 4, 10, 11]
    prompt_len = 4
    labels = build_masked_labels(prompt_len, full, pad_token_id=0)
    assert labels[:4] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
    assert labels[4:] == [10, 11]


def test_padding_masked():
    full = [1, 2, 3, 0, 0]
    labels = build_masked_labels(2, full, pad_token_id=0)
    assert labels[0] == IGNORE_INDEX
    assert labels[1] == IGNORE_INDEX
    assert labels[2] == 3
    assert labels[3] == IGNORE_INDEX
    assert labels[4] == IGNORE_INDEX
