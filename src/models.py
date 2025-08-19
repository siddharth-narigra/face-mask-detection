"""
Model loading and configuration module for Face Mask Detection
"""
import streamlit as st
import torch
import os
from typing import Any


@st.cache_resource
def load_model(weights_path: str) -> Any:
    """Load YOLOv5 model with Windows compatibility fixes"""
    try:
        # Convert to absolute path to avoid path issues
        abs_path = os.path.abspath(weights_path)
        
        # Load model with Windows compatibility fixes
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=abs_path,
            force_reload=True,
            trust_repo=True
        )
        setattr(model, 'conf', 0.4)  # Set confidence threshold to 40%
        return model
            
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def setup_environment():
    """Setup environment variables for Windows compatibility"""
    # Set environment variables for Windows compatibility
    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
    os.environ['YOLOv5_VERBOSE'] = 'False'
    
    # Fix for Windows PosixPath issue
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
