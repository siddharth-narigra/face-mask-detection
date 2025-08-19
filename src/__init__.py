"""
Face Mask Detection - Source Package

This package contains all the core modules for the Face Mask Detection application.
"""

__version__ = "1.0.0"
__author__ = "Face Mask Detection Team"

# Import main components for easy access
from .models import load_model, setup_environment
from .ui_components import apply_custom_css, render_header, render_footer
from .video_processor import process_video, BoxSmoother
from .image_processor import process_image_tab
from .webcam_processor import process_webcam_tab

__all__ = [
    "load_model",
    "setup_environment", 
    "apply_custom_css",
    "render_header",
    "render_footer",
    "process_video",
    "BoxSmoother",
    "process_image_tab",
    "process_webcam_tab"
]
