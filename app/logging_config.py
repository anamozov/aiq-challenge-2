"""
Centralized logging configuration for the AIQ Challenge application.
"""
import os
import logging
import logging.handlers
from pathlib import Path


def setup_logging(name=None, level=None, log_file=None):
    """
    Setup centralized logging configuration.
    
    Args:
        name: Logger name (defaults to calling module)
        level: Logging level (defaults to INFO)
        log_file: Optional log file path
    """
    # Get log level from environment or use default
    if level is None:
        level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
    
    # Create logger
    logger = logging.getLogger(name or __name__)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Create formatter with more context
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified or if logs directory exists
    if log_file or Path('logs').exists():
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_file or log_dir / f'{name or "app"}.log'
        
        # Rotating file handler to prevent large log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """Get logger with standard configuration."""
    return setup_logging(name)


# Test logger for test files
def get_test_logger(name):
    """Get logger specifically configured for test files."""
    logger = setup_logging(name, level=logging.INFO)
    return logger
