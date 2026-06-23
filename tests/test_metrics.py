from src.metrics import accuracy, expected_calibration_error


def test_accuracy() -> None:
    assert accuracy([0, 1, 1, 0], [0, 0, 1, 0]) == 0.75


def test_expected_calibration_error_perfect_bin() -> None:
    value = expected_calibration_error([1.0, 1.0], [True, True], n_bins=2)
    assert value == 0.0
