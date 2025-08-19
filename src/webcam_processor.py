# """
# Webcam processing module for Face Mask Detection using streamlit-webrtc
# """
# import streamlit as st
# import cv2
# import numpy as np
# from typing import Any
# import av
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# from .video_processor import process_video_frame, BoxSmoother


# class MaskDetectionTransformer(VideoTransformerBase):
#     """Video transformer for real-time mask detection using streamlit-webrtc"""
    
#     def __init__(self):
#         super().__init__()
#         self.model = None
#         self.smoother = BoxSmoother(smoothing_method="ema", alpha=0.5, iou_threshold=0.6)
    
#     def set_model(self, model):
#         """Set the YOLO model for detection"""
#         self.model = model
    
#     def transform(self, frame):
#         """Transform each frame with mask detection"""
#         if self.model is None:
#             return frame
            
#         # Convert frame to numpy array (BGR)
#         img = frame.to_ndarray(format="bgr24")
        
#         # Run YOLO detection + smoothing
#         processed_img = process_video_frame(self.model, img, self.smoother)
        
#         # Return the processed numpy array directly
#         return processed_img


# def process_webcam_tab(model: Any):
#     """Render and handle the webcam processing tab with real-time detection using WebRTC"""
#     st.markdown('<div class="section-header">🎥 Real-time Webcam Detection</div>', unsafe_allow_html=True)
    
#     st.write("Live YOLO-based mask detection running in the browser.")
    
#     # Information and instructions
#     st.markdown("""
#     <div class="info-box">
#         <h4>📹 Real-time Webcam Detection</h4>
#         <p>This uses WebRTC for smooth, real-time video streaming with live mask detection.</p>
#         <p><strong>Features:</strong></p>
#         <ul>
#             <li>🎯 Live webcam feed inside Streamlit</li>
#             <li>📊 EMA smoothing (α=0.5) for stable bounding boxes</li>
#             <li>⚡ Real-time YOLO detection</li>
#             <li>🌐 Works in any modern browser</li>
#         </ul>
#         <p><strong>Instructions:</strong></p>
#         <ul>
#             <li>Click "START" to begin webcam streaming</li>
#             <li>Allow camera permissions when prompted by your browser</li>
#             <li>Detection runs automatically on the live video feed</li>
#             <li>Click "STOP" to end the stream</li>
#         </ul>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Detection Legend
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown("🟢 **With Mask** (mask)")
#     with col2:
#         st.markdown("🔴 **Without Mask** (without_mask)")
#     with col3:
#         st.markdown("🔵 **Incorrect Mask** (incorrect_mask)")
    
#     st.markdown("---")
    
#     # Create transformer factory that includes the model
#     def transformer_factory():
#         transformer = MaskDetectionTransformer()
#         transformer.set_model(model)
#         return transformer
    
#     # WebRTC Streamer with proper configuration
#     webrtc_ctx = webrtc_streamer(
#         key="mask-detection",
#         video_processor_factory=transformer_factory,
#         media_stream_constraints={
#             "video": {
#                 "width": {"ideal": 640},
#                 "height": {"ideal": 480},
#                 "frameRate": {"ideal": 15}
#             },
#             "audio": False
#         },
#         async_processing=True,
#     )
    
#     # Status information
#     if webrtc_ctx.state.playing:
#         st.success("🔴 **Live Detection Active** - Real-time mask detection is running!")
#     elif webrtc_ctx.state.signalling:
#         st.info("🔄 **Connecting...** - Establishing webcam connection")
#     else:
#         st.info("⚫ **Webcam Stopped** - Click START to begin detection")
    
#     # Troubleshooting section
#     st.markdown("---")
#     with st.expander("🔧 Troubleshooting"):
#         st.markdown("""
#         **If the webcam is not working:**
        
#         1. **Browser Permissions**: Make sure your browser allows camera access
#         2. **HTTPS Required**: WebRTC requires HTTPS in production (works on localhost)
#         3. **Camera Availability**: Ensure no other application is using your camera
#         4. **Browser Support**: Use Chrome, Firefox, or Edge (latest versions)
#         5. **Firewall**: Check if your firewall is blocking WebRTC connections
        
#         **Supported Browsers:**
#         - ✅ Google Chrome (recommended)
#         - ✅ Mozilla Firefox  
#         - ✅ Microsoft Edge
#         - ❌ Safari (limited WebRTC support)
#         """)
    
#     # Alternative: Single frame capture for fallback
#     st.markdown("---")
#     st.markdown("### 📸 Alternative: Single Frame Analysis")
#     st.markdown("*For testing purposes or if live feed doesn't work*")
    
#     if st.button("Capture & Analyze One Frame", use_container_width=True):
#         try:
#             cap = cv2.VideoCapture(0)
#             if cap.isOpened():
#                 ret, frame = cap.read()
#                 if ret:
#                     # Process single frame
#                     single_smoother = BoxSmoother(smoothing_method="ema", alpha=0.5, iou_threshold=0.6)
#                     processed_frame = process_video_frame(model, frame, single_smoother)
                    
#                     # Convert and display
#                     processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
#                     st.image(processed_frame_rgb, caption="Single Frame Detection Result", use_container_width=True)
#                     st.success("✅ Frame processed successfully!")
#                 else:
#                     st.error("❌ Failed to capture frame")
#                 cap.release()
#             else:
#                 st.error("❌ Cannot access webcam")
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")



"""
Webcam processing module for Face Mask Detection using streamlit-webrtc
"""
import streamlit as st
import cv2
import numpy as np
from typing import Any
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from .video_processor import process_video_frame, BoxSmoother


class MaskDetectionTransformer(VideoTransformerBase):
    """Video transformer for real-time mask detection using streamlit-webrtc"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.smoother = BoxSmoother(
            smoothing_method="ema", alpha=0.5, iou_threshold=0.6
        )

    def set_model(self, model):
        """Set the YOLO model for detection"""
        self.model = model

    def transform(self, frame):
        """Transform each frame with mask detection"""
        if self.model is None:
            return frame

        # Step 1: Convert to numpy (BGR format, same as video_processor input)
        img = frame.to_ndarray(format="bgr24")

        # ✅ Step 2: Match preprocessing from video_processor
        # (resize only if video_processor does resizing, else leave as is)
        # Example: YOLO usually auto-resizes, so don't resize unless necessary
        # img = cv2.resize(img, (640, 640))

        # Step 3: Run YOLO detection + smoothing
        processed_img = process_video_frame(self.model, img, self.smoother)

        # Step 4: Return processed frame (still BGR, streamlit-webrtc expects numpy)
        return processed_img


def process_webcam_tab(model: Any):
    """Render and handle the webcam processing tab with real-time detection using WebRTC"""
    st.markdown('<div class="section-header">Real-time Webcam Detection</div>', unsafe_allow_html=True)

    st.write("Live YOLO-based mask detection running in the browser.")





    # Create transformer factory that includes the model
    def transformer_factory():
        transformer = MaskDetectionTransformer()
        transformer.set_model(model)
        return transformer

    # WebRTC Streamer with proper configuration
    webrtc_ctx = webrtc_streamer(
        key="mask-detection",
        video_processor_factory=transformer_factory,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}
            },
            "audio": False
        },
        async_processing=True,
    )

    # Status information
    if webrtc_ctx.state.playing:
        st.success("**Live Detection Active** - Real-time mask detection is running!")
    elif webrtc_ctx.state.signalling:
        st.info("**Connecting...** - Establishing webcam connection")
    else:
        st.info("**Webcam Stopped** - Click START to begin detection")

    # Troubleshooting section
    with st.expander("Troubleshooting"):
        st.markdown("""
        **If the webcam is not working:**
        
        1. **Browser Permissions**: Make sure your browser allows camera access  
        2. **HTTPS Required**: WebRTC requires HTTPS in production (works on localhost)  
        3. **Camera Availability**: Ensure no other application is using your camera  
        4. **Browser Support**: Use Chrome, Firefox, or Edge (latest versions)  
        5. **Firewall**: Check if your firewall is blocking WebRTC connections  

        **Supported Browsers:**
        - Google Chrome (recommended)  
        - Mozilla Firefox  
        - Microsoft Edge  
        - Safari (limited WebRTC support)
        """)


