"""
Image processing module for Face Mask Detection
"""
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from typing import Any, List, Tuple
import numpy as np
from .ui_components import render_detection_badge, get_detection_display_info


def process_image_tab(model: Any):
    """Render and handle the image processing tab"""
    # Initialize session state for uploaded image
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
        st.session_state.image_results = None
    if 'uploader_key_counter' not in st.session_state:
        st.session_state.uploader_key_counter = 0
    
    st.markdown('<div class="section-header">Image Upload & Analysis</div>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown("""
    <div class="upload-zone">
        <h4>Drop your image here</h4>
        <p>Drag and drop an image file, or click browse to select</p>
        <p style="font-size: 0.85rem; color: #64748b;">Supported formats: JPG, JPEG, PNG • Max size: 10MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader (full width to match upload zone)
    # Use a counter-based key to force widget reset when cleared
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"],
        key=f"image_uploader_{st.session_state.uploader_key_counter}"
    )
    

    
    # Process uploaded file
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file
        image = Image.open(uploaded_file).convert("RGB")
        
        # Display results in two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### Detection Results")
            
            with st.spinner("Analyzing image..."):
                results = model(image)
                corrected_image = draw_corrected_detections(image, results)
                st.session_state.image_results = results

            st.image(corrected_image, use_container_width=True)
        
        # Detection summary
        st.markdown('<div class="section-header">Detection Summary</div>', unsafe_allow_html=True)
        
        detections = results.xyxy[0].tolist()
        
        if detections:
            # Create detection cards
            cols = st.columns(min(len(detections), 3))
            for i, (*box, conf, cls) in enumerate(detections):
                class_name = results.names[int(cls)]
                display_name, badge_class, icon = get_detection_display_info(class_name)
                
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="detection-card">
                        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                            <strong style="color: #1e293b;">{display_name}</strong>
                        </div>
                        <div class="status-badge {badge_class}">
                            Confidence: {conf:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h5>No faces detected</h5>
                <p>Try uploading an image with clear, visible faces for better detection results.</p>
            </div>
            """, unsafe_allow_html=True)
    



def get_detection_color(class_name: str) -> Tuple[int, int, int]:
    """Get RGB color for bounding box based on class name (for PIL)"""
    class_lower = class_name.lower()
    if 'without_mask' in class_lower or 'no_mask' in class_lower:
        return (0, 255, 0)  # Green - these are actually people WITH masks
    elif 'with_mask' in class_lower or 'mask' in class_lower:
        return (255, 0, 0)  # Red - these are actually people WITHOUT masks  
    elif 'incorrect' in class_lower or 'improper' in class_lower or 'wrong' in class_lower:
        return (0, 135, 255)  # Sky blue for incorrect mask
    else:
        return (255, 0, 0)  # Default to red for unknown classes


def draw_corrected_detections(image: Image.Image, results) -> Image.Image:
    """Draw corrected bounding boxes on image with proper labels"""
    # Create a copy of the image to draw on
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Get detections
    detections = results.xyxy[0].tolist()
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        class_name = results.names[int(cls)]
        
        # Get corrected display name and color
        display_name, _, _ = get_detection_display_info(class_name)
        color = get_detection_color(class_name)
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background and text
        label = f"{display_name}: {conf:.2f}"
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        label_y = max(text_height + 5, y1 - text_height - 5)
        draw.rectangle([x1, label_y - text_height - 5, x1 + text_width + 10, label_y + 5], fill=color)
        
        # Draw label text
        draw.text((x1 + 5, label_y - text_height), label, fill=(255, 255, 255), font=font)
    
    return img_copy


def get_image_detections(model: Any, image: Image.Image) -> List[Tuple]:
    """Process image and return detection results"""
    results = model(image)
    return results.xyxy[0].tolist()
