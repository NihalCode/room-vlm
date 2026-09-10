"""Candidate scoring unit tests (no model download)."""

from room_vlm.model.scoring import score_labels_from_token_logprobs, softmax_dict


def test_softmax_dict():
    probs = softmax_dict({"a": 0.0, "b": 0.0})
    assert abs(probs["a"] - 0.5) < 1e-6
    assert abs(probs["b"] - 0.5) < 1e-6


def test_length_normalized_scoring():
    # "living room" has more tokens; length norm should not automatically lose to longer sequences of similar avg.
    result = score_labels_from_token_logprobs(
        {
            "bedroom": [-0.1, -0.1],
            "living room": [-0.1, -0.1, -0.1],
            "bathroom": [-2.0],
            "kitchen": [-3.0],
        },
        length_normalize=True,
    )
    assert result["label"] in {"bedroom", "living room"}
    assert abs(sum(result["scores"].values()) - 1.0) < 1e-6
    assert result["confidence"] == max(result["scores"].values())
