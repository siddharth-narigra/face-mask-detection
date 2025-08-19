import streamlit as st
import torch
from PIL import Image
from typing import Any, Dict, List, Tuple
import os
import pathlib
import cv2
import numpy as np
import tempfile
from collections import defaultdict, deque

# Set environment variables for Windows compatibility
os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
os.environ['YOLOv5_VERBOSE'] = 'False'

# Fix for Windows PosixPath issue
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Safe globals handling removed - not needed for this YOLOv5 implementation

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
        
        class_name = results.names[int(cls)]
        
        # Choose color based on class (BGR format for OpenCV)
        # Fix inverted class labels - swap the logic
        class_lower = class_name.lower()
        if 'without_mask' in class_lower or 'no_mask' in class_lower:
            color = (0, 255, 0)  # Parrot green - these are actually people WITH masks
        elif 'with_mask' in class_lower or 'mask' in class_lower:
            color = (0, 0, 255)  # Red - these are actually people WITHOUT masks  
        elif 'incorrect' in class_lower or 'improper' in class_lower or 'wrong' in class_lower:
            color = (255, 135, 0)  # Sky blue for incorrect mask
        else:
            color = (0, 0, 255)  # Default to red for unknown classes
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{class_name}: {conf:.2f}"
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

def process_video(model: Any, input_path: str, output_path: str, progress_callback=None) -> bool:
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

st.set_page_config(
    page_title="Face Mask Detection", 
    layout="wide",
    page_icon="😷",
    initial_sidebar_state="collapsed"
)

# --- Load model once and cache it ---
@st.cache_resource
def load_model(weights_path: str) -> Any:
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

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: transparent;
        border-radius: 8px;
        color: #666;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        background: #f0f2f6;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .badge-mask { background: #d4edda; color: #155724; }
    .badge-no-mask { background: #f8d7da; color: #721c24; }
    .badge-incorrect { background: #cce7ff; color: #004085; }
</style>
""", unsafe_allow_html=True)

# Modern header
st.markdown("""
<div class="main-header">
    <h1>😷 Face Mask Detection</h1>
    <p>Advanced AI-powered mask compliance monitoring with YOLOv5</p>
</div>
""", unsafe_allow_html=True)

# Load your weights
MODEL = load_model("weights/best.pt")

if MODEL is None:
    st.error("🚨 Model failed to load. Please check your weights file.")
    st.stop()

# Create tabs with modern styling
tab1, tab2, tab3 = st.tabs(["📸 Image Analysis", "🎥 Video Processing", "📹 Webcam Stream"])

with tab1:
    st.markdown("### 📸 Upload an Image for Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h4>📁 Select Image File</h4>
            <p>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Original Image", use_container_width=True)

    with col2:
        if uploaded_file is not None:
            with st.spinner("🔍 Analyzing image..."):
        results = MODEL(image)
        rendered_rgb = results.render()[0]

            st.image(rendered_rgb, caption="🎯 Detection Results", use_container_width=True)

            # Modern detection results
            st.markdown("### 📊 Detection Summary")
            detections = results.xyxy[0].tolist()
            
            if detections:
                for *box, conf, cls in detections:
        class_name = results.names[int(cls)]
                    class_lower = class_name.lower()
                    
                    if 'without_mask' in class_lower or 'no_mask' in class_lower:
                        display_name = "With Mask"
                        badge_class = "badge-mask"
                        icon = "✅"
                    elif 'with_mask' in class_lower:
                        display_name = "Without Mask"
                        badge_class = "badge-no-mask"
                        icon = "❌"
                    else:
                        display_name = "Incorrect Mask"
                        badge_class = "badge-incorrect"
                        icon = "⚠️"
                    
                    st.markdown(f"""
                    <div class="feature-card">
                        {icon} <span class="status-badge {badge_class}">{display_name}</span>
                        <br><small>Confidence: {conf:.1%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No faces detected in the image.")

with tab2:
    st.markdown("### 🎥 Video Processing with AI Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="upload-section">
            <h4>📁 Upload Video File</h4>
            <p>Supported formats: MP4, AVI, MOV</p>
            <small>⚡ Features smooth bounding box tracking</small>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")
        
        if uploaded_video is not None:
            st.markdown("#### 📹 Original Video")
            st.video(uploaded_video)
            
            # Processing controls
            st.markdown("#### 🎛️ Processing Controls")
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                process_btn = st.button("🚀 Process Video", type="primary", use_container_width=True)
            with col_btn2:
                st.markdown("""
                <div class="feature-card" style="margin: 0; padding: 0.5rem;">
                    <small>✨ EMA Smoothing<br>🎯 Object Tracking<br>🎨 Color Coding</small>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_video is not None:
            if process_btn:
                # Save uploaded video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                    tmp_input.write(uploaded_video.read())
                    input_path = tmp_input.name
                
                # Create output path
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                    output_path = tmp_output.name
                
                # Processing UI
                st.markdown("#### 🔄 Processing Status")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.markdown(f"**Processing... {progress*100:.1f}%** 🎬")
                
                with st.spinner("🎭 Applying AI detection and smoothing..."):
                    success = process_video(MODEL, input_path, output_path, update_progress)
                
                if success:
                    st.success("🎉 Video processed successfully!")
                    
                    # Show processed video
                    st.markdown("#### 🎯 Processed Video")
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    # Download section
                    st.markdown("#### 💾 Download Results")
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_bytes,
                        file_name="mask_detection_output.mp4",
                        mime="video/mp4",
                        type="primary",
                        use_container_width=True
                    )
                    
                    # Processing info
                    st.markdown("""
                    <div class="feature-card">
                        <h5>✨ Processing Features Applied:</h5>
                        <ul>
                            <li>🎯 Real-time mask detection</li>
                            <li>🔄 Exponential moving average smoothing</li>
                            <li>👥 Object tracking with persistent IDs</li>
                            <li>🎨 Color-coded bounding boxes</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to process video. Please try again.")
                
                # Cleanup temp files
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass

with tab3:
    st.markdown("### 📹 Real-time Webcam Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🎥 Live Camera Stream</h4>
            <p>Real-time mask detection using your device camera</p>
            <br>
            <h5>📋 Requirements:</h5>
            <ul>
                <li>🖥️ Local environment execution</li>
                <li>📷 Camera permissions enabled</li>
                <li>🐍 Python with OpenCV installed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h5>🚀 Quick Start Guide:</h5>
            <ol>
                <li>Run the app locally</li>
                <li>Grant camera permissions</li>
                <li>Click "Start Webcam"</li>
                <li>Position yourself in view</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="upload-section">
            <h4>💻 Local Development Code</h4>
            <p>Copy this code to run webcam detection locally</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.code("""
import cv2
import streamlit as st

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create Streamlit placeholder for video
video_placeholder = st.empty()

# Process frames in real-time
while True:
    ret, frame = cap.read()
    if ret:
        # Process with your MODEL
        processed_frame = process_video_frame(MODEL, frame, smoother)
        
        # Display in Streamlit
        video_placeholder.image(processed_frame, channels="BGR")
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
        """, language="python")
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("📷 Start Webcam", type="primary", use_container_width=True):
                st.warning("⚠️ This feature requires local execution with camera permissions.")
        
        with col_btn2:
            st.markdown("""
            <div class="feature-card" style="margin: 0; padding: 0.5rem;">
                <small>🔴 Live Detection<br>⚡ Real-time Processing<br>🎯 Instant Results</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("💡 **Tip:** For best results, ensure good lighting and position your face clearly in the camera view.")

# Modern footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>🚀 AI-Powered Face Mask Detection</h4>
    <p>Built with <strong>YOLOv5</strong> • <strong>Streamlit</strong> • <strong>OpenCV</strong></p>
    <div style="margin-top: 1rem;">
        <span class="status-badge badge-mask">✅ With Mask</span>
        <span class="status-badge badge-no-mask">❌ Without Mask</span>
        <span class="status-badge badge-incorrect">⚠️ Incorrect Mask</span>
    </div>
    <br>
    <small>🎯 Real-time Detection • 🔄 Smooth Tracking • 📊 High Accuracy</small>
</div>
""", unsafe_allow_html=True)
