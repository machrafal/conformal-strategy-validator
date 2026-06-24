"""tests/test_report.py - unit tests for csv_validator/report.py"""

from csv_validator.report import ValidationReport, validate


def test_validate_returns_validation_report(real_like_returns):
    report = validate(real_like_returns, n_trials=10)
    assert isinstance(report, ValidationReport)


def test_report_sharpe_is_float(real_like_returns):
    report = validate(real_like_returns, n_trials=10)
    assert isinstance(report.sharpe, float)


def test_report_summary_contains_header(real_like_returns):
    report = validate(real_like_returns, n_trials=10)
    assert "Validation Report" in report.summary()
