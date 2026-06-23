from src.models.score import score_options_from_log_likelihoods, softmax


def test_softmax_prefers_higher_score() -> None:
    probs = softmax([2.0, 0.0])
    assert probs[0] > probs[1]
    assert abs(sum(probs) - 1.0) < 1e-9


def test_length_normalized_option_scoring() -> None:
    scores = score_options_from_log_likelihoods(
        ["short", "long"],
        [-1.0, -2.0],
        [1, 4],
        length_normalized=True,
    )
    by_option = {score.option: score for score in scores}
    assert by_option["long"].probability > by_option["short"].probability
