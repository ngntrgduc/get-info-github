"""Just a logger"""

import logging

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.FileHandler('runtime.log', encoding='utf-8')
handler.setFormatter(
    logging.Formatter(
        fmt     = '{asctime} | {levelname:8s}  {message}',
        datefmt = '%D %T',
        style   = '{'
    )
)
root.addHandler(handler)
logger = logging.getLogger(__name__)

def debug(message: str):
    """Log debug to file"""
    logging.debug(message)

def info(message: str):
    """Log message to file"""
    logging.info(message)

def warning(message: str):
    """Log warning to file"""
    logging.warning(message)

def error(message: str):
    """Log error to file"""
    logging.error(message)

def exception(message: str):
    """Log exception to file"""
    logging.exception(message)

def critical(message: str):
    """Log critical to file"""
    logging.critical(message)
