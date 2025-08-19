"""
Video processing module with advanced bounding box smoothing
"""
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Callable
from collections import defaultdict, deque


class BoxSmoother:
    """Advanced bounding box smoother with EMA and object tracking"""
    def __init__(self, smoothing_method: str = "ema", alpha: float = 0.6, iou_threshold: float = 0.5):
        self.smoothing_method = smoothing_method  # "ema", "moving_average", or "hybrid"
        self.alpha = alpha  # EMA weight factor (0.0-1.0, higher = more responsive)
        self.iou_threshold = iou_threshold  # IoU threshold for object matching
        
        # Storage for different smoothing methods
        self.ema_boxes: Dict[int, List[float]] = {}  # EMA smoothed boxes
        self.box_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=5))  # Moving average
        self.object_tracks: Dict[int, Dict] = {}  # Object tracking info
        self.next_track_id = 0
        self.max_disappeared_frames = 10
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections_to_tracks(self, detections: List[Tuple]) -> Dict[int, Tuple]:
        """Match current detections to existing tracks using IoU"""
        matches = {}
        used_detections = set()
        
        # Try to match each existing track with a detection
        for track_id, track_info in self.object_tracks.items():
            best_match_idx = -1
            best_iou = 0.0
            
            for i, (*box, conf, cls) in enumerate(detections):
                if i in used_detections:
                    continue
                
                iou = self.calculate_iou(track_info['last_box'], box)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx >= 0:
                matches[track_id] = detections[best_match_idx]
                used_detections.add(best_match_idx)
                self.object_tracks[track_id]['disappeared_frames'] = 0
            else:
                self.object_tracks[track_id]['disappeared_frames'] += 1
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                matches[track_id] = detection
                
                box = detection[:4]
                self.object_tracks[track_id] = {
                    'last_box': box,
                    'disappeared_frames': 0,
                    'class': detection[5]
                }
        
        # Remove tracks that have disappeared for too long
        tracks_to_remove = []
        for track_id, track_info in self.object_tracks.items():
            if track_info['disappeared_frames'] > self.max_disappeared_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.object_tracks[track_id]
            if track_id in self.ema_boxes:
                del self.ema_boxes[track_id]
            if track_id in self.box_history:
                del self.box_history[track_id]
        
        return matches
    
    def smooth_boxes(self, detections: List[Tuple]) -> List[Tuple]:
        """Apply advanced smoothing to bounding boxes"""
        if not detections:
            return []
        
        # Match detections to existing tracks
        matched_detections = self.match_detections_to_tracks(detections)
        smoothed = []
        
        for track_id, (*box, conf, cls) in matched_detections.items():
            if self.smoothing_method == "ema":
                smoothed_box = self._apply_ema_smoothing(track_id, box)
            elif self.smoothing_method == "moving_average":
                smoothed_box = self._apply_moving_average_smoothing(track_id, box)
            else:  # hybrid
                smoothed_box = self._apply_hybrid_smoothing(track_id, box)
            
            # Update track info
            self.object_tracks[track_id]['last_box'] = smoothed_box
            smoothed.append((*smoothed_box, conf, cls))
        
        return smoothed
    
    def _apply_ema_smoothing(self, track_id: int, box: List[float]) -> List[float]:
        """Apply Exponential Moving Average smoothing"""
        if track_id not in self.ema_boxes:
            # First detection for this track
            self.ema_boxes[track_id] = list(box)
            return list(box)
        
        # Apply EMA: smoothed = α * current + (1-α) * previous
        smoothed_box = []
        for i in range(4):
            smoothed_coord = self.alpha * box[i] + (1 - self.alpha) * self.ema_boxes[track_id][i]
            smoothed_box.append(smoothed_coord)
        
        self.ema_boxes[track_id] = smoothed_box
        return smoothed_box
    
    def _apply_moving_average_smoothing(self, track_id: int, box: List[float]) -> List[float]:
        """Apply moving average smoothing"""
        self.box_history[track_id].append(box)
        
        if len(self.box_history[track_id]) <= 1:
            return list(box)
        
        # Calculate average over history
        history = list(self.box_history[track_id])
        smoothed_box = [
            sum(coord[i] for coord in history) / len(history)
            for i in range(4)
        ]
        return smoothed_box
    
    def _apply_hybrid_smoothing(self, track_id: int, box: List[float]) -> List[float]:
        """Apply hybrid smoothing (EMA + moving average)"""
        # Use EMA for quick response, then apply light moving average
        ema_box = self._apply_ema_smoothing(track_id, box)
        return self._apply_moving_average_smoothing(track_id, ema_box)


def fix_label_mapping(class_name: str) -> str:
    """Fix the inverted label mapping from the model"""
    class_lower = class_name.lower()
    if 'incorrect' in class_lower or 'improper' in class_lower or 'wrong' in class_lower:
        # Keep incorrect as is (check this first before other mask checks)
        return "incorrect_mask"
    elif 'mask' in class_lower and 'without' not in class_lower:
        # Model says "mask" but it's actually "without_mask"
        return "without_mask"
    elif 'without_mask' in class_lower or 'no_mask' in class_lower:
        # Model says "without_mask" but it's actually "mask"
        return "mask"
    else:
        # Default - assume it's without mask
        return "without_mask"


def get_detection_color(class_name: str) -> Tuple[int, int, int]:
    """Get color for bounding box based on corrected class name"""
    class_lower = class_name.lower()
    if 'mask' in class_lower and 'without' not in class_lower and 'incorrect' not in class_lower:
        return (0, 255, 0)  # Green for WITH mask
    elif 'without_mask' in class_lower or 'no_mask' in class_lower:
        return (0, 0, 255)  # Red for WITHOUT mask
    elif 'incorrect' in class_lower or 'improper' in class_lower or 'wrong' in class_lower:
        return (255, 0, 0)  # Blue for incorrect mask
    else:
        return (0, 0, 255)  # Default to red for unknown classes


def apply_nms(detections: List[Tuple], nms_threshold: float = 0.4) -> List[Tuple]:
    """Apply Non-Maximum Suppression to remove duplicate boxes"""
    if not detections:
        return []
    
    # Extract boxes and scores
    boxes = np.array([det[:4] for det in detections])
    scores = np.array([det[4] for det in detections])
    
    # Apply NMS using OpenCV
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), 
        scores.tolist(), 
        score_threshold=0.3,  # Minimum confidence
        nms_threshold=nms_threshold
    )
    
    # Return filtered detections
    if len(indices) > 0:
        # Convert to a flat list safely
        indices = np.array(indices).flatten().tolist()
        return [detections[i] for i in indices]
    return []


def process_video_frame(model: Any, frame: np.ndarray, smoother: BoxSmoother) -> np.ndarray:
    """Process a single video frame with YOLO detection and smooth bounding boxes"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Convert BGR to RGB for YOLO (YOLO expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(rgb_frame)
    
    # Get detections in xyxy format (x1, y1, x2, y2, confidence, class)
    detections = results.xyxy[0].cpu().numpy().tolist()
    
    # Apply NMS to remove duplicate boxes
    detections = apply_nms(detections, nms_threshold=0.4)
    
    # Apply smoothing to detections
    smoothed_detections = smoother.smooth_boxes(detections)
    
    # Draw smoothed bounding boxes on the original BGR frame
    for *box, conf, cls in smoothed_detections:
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure coordinates are valid and within bounds
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width - 1))
        y2 = max(y1 + 1, min(y2, height - 1))
        
        original_class_name = results.names[int(cls)]
        # Fix the inverted label mapping
        corrected_class_name = fix_label_mapping(original_class_name)
        color = get_detection_color(corrected_class_name)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background (use corrected label)
        label = f"{corrected_class_name}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Position label above the box, but within frame
        label_y = max(label_size[1] + 10, y1 - 5)
        label_x1 = x1
        label_x2 = min(x1 + label_size[0] + 5, width - 1)
        label_y1 = label_y - label_size[1] - 5
        label_y2 = label_y + 5
        
        # Draw label background
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 2, label_y - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame


def process_video(model: Any, input_path: str, output_path: str, progress_callback: Optional[Callable] = None) -> bool:
    """Process video file with YOLO detection and smooth bounding boxes"""
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer with fallback codecs for browser compatibility
    codecs_to_try = ['H264', 'XVID', 'MJPG', 'mp4v']
    out = None
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter.fourcc(*codec)  # type: ignore
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                break
            out.release()
        except:
            continue
    
    if out is None or not out.isOpened():
        # Final fallback with default codec
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # type: ignore
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize advanced smoother with EMA
    smoother = BoxSmoother(smoothing_method="ema", alpha=0.7, iou_threshold=0.5)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = process_video_frame(model, frame, smoother)
            
            # Write frame
            out.write(processed_frame)
            
            frame_count += 1
            
            # Update progress
            if progress_callback and total_frames > 0:
                progress = frame_count / total_frames
                progress_callback(progress)
    
    finally:
        cap.release()
        out.release()
    
    return True
