import logging
import sys
from src.utils.config import LOG_LEVEL

def setup_logger(name: str = "credit_risk_platform") -> logging.Logger:
    """Configures structured logger for console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

logger = setup_logger()
