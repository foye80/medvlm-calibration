from src.calibrate import apply_temperature, fit_temperature, negative_log_likelihood


def test_smoke_returns_identity() -> None:
    fit = fit_temperature([[1.0, 0.0]], [0], smoke=True)
    assert fit.temperature == 1.0


def test_apply_temperature_softens() -> None:
    sharp = apply_temperature([2.0, 0.0], 1.0)
    soft = apply_temperature([2.0, 0.0], 5.0)
    # Higher temperature moves the distribution toward uniform.
    assert max(soft) < max(sharp)


def test_fit_softens_overconfident_scores() -> None:
    # Argmax is class 0 with a large margin, but only 60% are actually class 0.
    scores = [[4.0, 0.0]] * 6 + [[4.0, 0.0]] * 4
    labels = [0] * 6 + [1] * 4
    fit = fit_temperature(scores, labels)
    # Overconfidence should be corrected by softening (T > 1).
    assert fit.temperature > 1.0
    # The fitted temperature must not increase NLL relative to T=1.
    assert negative_log_likelihood(scores, labels, fit.temperature) <= negative_log_likelihood(
        scores, labels, 1.0
    )
