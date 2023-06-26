import logging

disable_loggers = ["botocore", "s3transfer", "asyncio", "PIL", "matplotlib"]


def pytest_configure():
    for logger_name in disable_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
