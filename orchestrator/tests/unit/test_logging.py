import structlog

from orchestrator.logging import setup_logging


def test_setup_logging_returns_logger():
    setup_logging(json_output=False)
    logger = structlog.get_logger("test")
    assert logger is not None


def test_logger_binds_context():
    setup_logging(json_output=False)
    logger = structlog.get_logger("test")
    bound = logger.bind(run_id="abc-123", symbol="BTC/USDT:USDT")
    assert bound is not None
