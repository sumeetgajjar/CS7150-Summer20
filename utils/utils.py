import logging
from logging import getLogger, Formatter, StreamHandler


class Utils:

    @classmethod
    def configure_logging(cls, level=logging.INFO, enable_logging_to_file=False, filepath=None):
        logger = getLogger()
        logger.setLevel(level)

        log_formatter = Formatter("[%(process)d] %(asctime)s [%(levelname)s] %(name)s: %(message)s")

        console_handler = StreamHandler()
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)

        if enable_logging_to_file:
            file_handler = logging.FileHandler(filepath)
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)
