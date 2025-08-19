"""
Configuration settings for Face Mask Detection application
"""

# Model Configuration
MODEL_WEIGHTS_PATH = "weights/best.pt"
CONFIDENCE_THRESHOLD = 0.4

# Video Processing Configuration
SMOOTHING_METHOD = "ema"  # "ema", "moving_average", or "hybrid"
EMA_ALPHA = 0.7  # Exponential Moving Average weight factor (0.0-1.0)
IOU_THRESHOLD = 0.5  # IoU threshold for object matching
MAX_DISAPPEARED_FRAMES = 10  # Max frames before removing a track
MOVING_AVERAGE_WINDOW = 5  # Window size for moving average

# Video Codec Configuration
VIDEO_CODECS = ['H264', 'XVID', 'MJPG', 'mp4v']

# UI Configuration
APP_TITLE = "Face Mask Detection"
APP_ICON = "😷"
PAGE_LAYOUT = "wide"

# Color Configuration (BGR format for OpenCV)
COLORS = {
    "with_mask": (0, 255, 0),      # Parrot green
    "without_mask": (0, 0, 255),    # Red  
    "incorrect_mask": (255, 135, 0) # Sky blue
}

# Detection Display Names
DISPLAY_NAMES = {
    "with_mask": "With Mask",
    "without_mask": "Without Mask", 
    "incorrect_mask": "Incorrect Mask"
}

# Environment Variables
ENV_VARS = {
    'TORCH_HOME': '~/.cache/torch',
    'YOLOv5_VERBOSE': 'False'
}
