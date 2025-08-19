"""
Face Mask Detection App - Main Application Entry Point
A modern, modular Streamlit application for AI-powered mask detection
"""
import streamlit as st
import tempfile
import os
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules from src
from src.models import load_model, setup_environment
from src.ui_components import apply_custom_css, render_header, render_footer
from src.image_processor import process_image_tab
from src.video_processor import process_video
from src.webcam_processor import process_webcam_tab
# Import configuration directly to avoid import issues
MODEL_WEIGHTS_PATH = "weights/best.pt"
APP_TITLE = "Face Mask Detection"


def setup_app():
    """Configure Streamlit app settings"""
    st.set_page_config(
        page_title=APP_TITLE, 
        layout="centered",
        initial_sidebar_state="collapsed"
    )


def process_video_tab(model, logger):
    """Render and handle the video processing tab"""
    st.markdown('<div class="section-header">Video Processing with AI Detection</div>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown("""
    <div class="upload-zone">
        <h4>Drop your video here</h4>
        <p>Drag and drop a video file, or click browse to select</p>
        <p style="font-size: 0.85rem; color: #64748b;">Supported formats: MP4, AVI, MOV • Max size: 100MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )
    
    if uploaded_video is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Original Video")
            st.video(uploaded_video)
            
            # Processing controls
            process_btn = st.button("Process Video", type="primary", use_container_width=True)
        
        with col2:
            if process_btn:
                # Save uploaded video to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                    tmp_input.write(uploaded_video.read())
                    input_path = tmp_input.name
                
                # Create output path
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
                    output_path = tmp_output.name
                
                # Log processing
                logger.info(f"Processing video: {uploaded_video.name}")
                
                # Processing UI
                st.markdown("#### Processing Status")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize progress tracking
                last_update_time = [0.0]  # Use list to make it mutable
                processing_complete = [False]  # Track completion state
                
                def update_progress(progress):
                    import time
                    current_time = time.time()
                    # Throttle updates to avoid overwhelming Streamlit (update every 100ms minimum)
                    if current_time - last_update_time[0] >= 0.1 or progress >= 1.0:
                        if not processing_complete[0]:  # Only update if not complete
                            progress_bar.progress(min(progress, 1.0))
                            status_text.markdown(f"**Processing... {progress*100:.1f}%**")
                            last_update_time[0] = current_time
                            
                            # Hide progress when complete
                            if progress >= 1.0:
                                processing_complete[0] = True
                                import time
                                time.sleep(0.5)  # Brief pause to show 100%
                                progress_bar.empty()
                                status_text.empty()
                
                with st.spinner("Applying AI detection and smoothing..."):
                    success = process_video(model, input_path, output_path, update_progress)
                
                if success:
                    st.success("Video processed successfully!")
                    logger.info("Video processing completed successfully")
                    
                    # Show processed video
                    st.markdown("#### Processed Video")
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    # Download section
                    st.markdown("#### Download Results")
                    st.download_button(
                        label="Download Processed Video",
                        data=video_bytes,
                        file_name="mask_detection_output.mp4",
                        mime="video/mp4",
                        type="primary",
                        use_container_width=True
                    )
                    

                else:
                    st.error("Failed to process video. Please try again.")
                    logger.error("Video processing failed")
                
                # Cleanup temp files
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass


def main():
    """Main application function"""
    # Setup
    setup_app()
    setup_environment()
    apply_custom_css()
    render_header()
    
    # Simple logging without utils import
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Face Mask Detection app started")
    
    # Load model
    MODEL = load_model(MODEL_WEIGHTS_PATH)
    
    if MODEL is None:
        st.error("🚨 Model failed to load. Please check your weights file.")
        logger.error(f"Failed to load model from {MODEL_WEIGHTS_PATH}")
        st.stop()
    
    logger.info(f"Model loaded successfully from {MODEL_WEIGHTS_PATH} with confidence 0.4")
    
    # Create tabs with improved navigation - reorder to make Image Upload first
    tab1, tab2, tab3 = st.tabs(["Image Upload", "Webcam Detection", "Video Processing"])
    
    with tab1:
        process_image_tab(MODEL)
    
    with tab2:
        process_webcam_tab(MODEL)
    
    with tab3:
        process_video_tab(MODEL, logger)
    
    logger.info("App rendering completed")


if __name__ == "__main__":
    main()