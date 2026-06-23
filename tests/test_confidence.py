from src.confidence import entropy, max_probability, parse_verbalized_confidence, self_consistency_confidence


def test_confidence_utilities() -> None:
    assert max_probability([0.2, 0.8]) == 0.8
    assert entropy([1.0, 0.0]) < 1e-6
    assert parse_verbalized_confidence("I am 87 percent confident.") == 0.87
    assert self_consistency_confidence(["yes", "yes", "no", "yes"]) == 0.75
