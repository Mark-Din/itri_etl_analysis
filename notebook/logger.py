import logging
import os

def init_log(log_name):
    logger = logging.getLogger(log_name)
    if not logger.handlers:
        os.makedirs('./logs', exist_ok=True)
        log_file = f'./logs/{log_name}.log'

        # File handler
        file_handler = logging.FileHandler(log_file, encoding='UTF-8')
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler (for console and cron)
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('%(levelname)s - %(message)s')
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        logger.setLevel(logging.INFO)
    return logger
