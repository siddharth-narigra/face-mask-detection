"""
Logging configuration for Face Mask Detection application
"""
import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "face_mask_detection",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure logger for the application
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


def log_model_info(logger: logging.Logger, model_path: str, confidence: float):
    """Log model loading information"""
    logger.info(f"Loading YOLO model from: {model_path}")
    logger.info(f"Confidence threshold set to: {confidence}")


def log_video_processing(logger: logging.Logger, input_path: str, output_path: str):
    """Log video processing information"""
    logger.info(f"Processing video: {input_path} -> {output_path}")


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log error with context"""
    error_msg = f"Error in {context}: {str(error)}" if context else f"Error: {str(error)}"
    logger.error(error_msg, exc_info=True)
