from naco.progressive_simpo import build_preference_pairs_from_session


def test_build_preference_pairs():
    session = {
        "session_id": "s1",
        "turns": [
            {"client": "hi", "counselor": "welcome"},
            {"client": "I am sad", "counselor": "I hear your sadness."},
        ],
    }
    pairs = build_preference_pairs_from_session(session)
    assert len(pairs) == 1
    assert "I hear your sadness." in pairs[0].chosen
