"""Tests for ``extract_landmarks_sorted``, duplicated across four scripts."""

import numpy as np
import pytest

from fakes import Hand, Landmark, Results, make_degenerate_hand, make_hand

SCRIPTS_WITH_EXTRACTOR = [
    "collect_data",
    "live_inference",
    "voxora_main",
    "voxora_main_backup",
]

FEATURES_PER_HAND = 63


@pytest.fixture(params=SCRIPTS_WITH_EXTRACTOR)
def extractor(request, load_script):
    return load_script(request.param).extract_landmarks_sorted


def _rising_hand(offset):
    """Landmarks running diagonally up and to the right."""
    return make_hand(offset=offset, scale=0.1)


def _falling_hand(offset):
    """Landmarks running down and to the right, so it is distinguishable."""
    return Hand([Landmark(offset + i * 0.1, offset - i * 0.1) for i in range(21)])


def test_no_hands_returns_zero_vector(extractor):
    features = extractor(Results(None))

    assert features.shape == (2 * FEATURES_PER_HAND,)
    assert not features.any()


def test_empty_hand_list_returns_zero_vector(extractor):
    features = extractor(Results([]))

    assert not features.any()


def test_single_hand_fills_first_slot_only(extractor):
    features = extractor(Results([_rising_hand(0.2)]))

    assert features[:FEATURES_PER_HAND].any()
    assert not features[FEATURES_PER_HAND:].any()


def test_wrist_is_the_origin_after_normalization(extractor):
    features = extractor(Results([_rising_hand(0.4)]))

    assert features[0:3] == pytest.approx([0.0, 0.0, 0.0])


def test_scale_normalizes_wrist_to_middle_mcp_distance_to_one(extractor):
    features = extractor(Results([_rising_hand(0.1)]))

    middle_mcp = features[9 * 3:9 * 3 + 3]
    assert np.linalg.norm(middle_mcp) == pytest.approx(1.0)


def test_features_are_translation_invariant(extractor):
    near = extractor(Results([_rising_hand(0.0)]))
    far = extractor(Results([_rising_hand(0.6)]))

    assert near == pytest.approx(far)


def test_features_are_scale_invariant(extractor):
    small = extractor(Results([make_hand(offset=0.1, scale=0.01)]))
    large = extractor(Results([make_hand(offset=0.1, scale=0.2)]))

    assert small == pytest.approx(large)


def test_hands_are_ordered_by_wrist_x_not_detection_order(extractor):
    left = _rising_hand(0.1)
    right = _falling_hand(0.7)

    in_order = extractor(Results([left, right]))
    reversed_order = extractor(Results([right, left]))

    assert in_order == pytest.approx(reversed_order)
    assert in_order[:FEATURES_PER_HAND] != pytest.approx(
        in_order[FEATURES_PER_HAND:]
    )


def test_only_the_two_leftmost_hands_are_used(extractor):
    hands = [_rising_hand(0.9), _rising_hand(0.1), _falling_hand(0.5)]

    features = extractor(Results(hands))
    two_leftmost = extractor(Results([hands[1], hands[2]]))

    assert features.shape == (2 * FEATURES_PER_HAND,)
    assert features == pytest.approx(two_leftmost)


def test_degenerate_hand_does_not_divide_by_zero(extractor):
    features = extractor(Results([make_degenerate_hand()]))

    assert np.isfinite(features).all()
    assert not features.any()
