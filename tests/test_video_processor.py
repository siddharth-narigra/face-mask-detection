"""
Unit tests for video_processor module
"""
import pytest
import numpy as np
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from video_processor import BoxSmoother, get_detection_color


class TestBoxSmoother:
    """Test cases for BoxSmoother class"""
    
    def test_initialization(self):
        """Test BoxSmoother initialization"""
        smoother = BoxSmoother()
        assert smoother.smoothing_method == "ema"
        assert smoother.alpha == 0.6
        assert smoother.iou_threshold == 0.5
    
    def test_calculate_iou(self):
        """Test IoU calculation"""
        smoother = BoxSmoother()
        
        # Test identical boxes
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        iou = smoother.calculate_iou(box1, box2)
        assert iou == 1.0
        
        # Test non-overlapping boxes
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        iou = smoother.calculate_iou(box1, box2)
        assert iou == 0.0
    
    def test_smooth_boxes_empty(self):
        """Test smoothing with empty detections"""
        smoother = BoxSmoother()
        result = smoother.smooth_boxes([])
        assert result == []


class TestDetectionColor:
    """Test cases for detection color function"""
    
    def test_with_mask_color(self):
        """Test color for with mask detection"""
        color = get_detection_color("without_mask")
        assert color == (0, 255, 0)  # Parrot green
    
    def test_without_mask_color(self):
        """Test color for without mask detection"""
        color = get_detection_color("with_mask")
        assert color == (0, 0, 255)  # Red
    
    def test_incorrect_mask_color(self):
        """Test color for incorrect mask detection"""
        color = get_detection_color("incorrect_mask")
        assert color == (255, 135, 0)  # Sky blue


if __name__ == "__main__":
    pytest.main([__file__])
