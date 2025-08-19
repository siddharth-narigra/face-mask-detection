"""
UI components and styling for Face Mask Detection app
"""
import streamlit as st


def apply_custom_css():
    """Apply modern professional CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
        /* Import professional font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styling */
        .main .block-container {
            font-family: 'Inter', sans-serif;
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            width: 100%;
            margin: 0 auto;
        }
        
        /* Consistent container system */
        .content-container {
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            box-sizing: border-box;
        }
        
        /* Fix text contrast on dark backgrounds */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: #ffffff !important;
            font-weight: 600;
        }
        
        .stMarkdown p {
            color: #e2e8f0 !important;
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            padding: 2rem 0 3rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
        }
        
        .main-header h1 {
            font-size: 3.2rem;
            font-weight: 700;
            margin: 0;
            color: white;
            letter-spacing: -0.02em;
        }
        
        .main-header p {
            font-size: 1.1rem;
            margin: 0.75rem 0 0 0;
            color: rgba(255, 255, 255, 0.9);
            font-weight: 400;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #f8fafc;
            border-radius: 12px;
            padding: 6px;
            margin-bottom: 2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            background: transparent;
            border-radius: 8px;
            color: #64748b;
            font-weight: 500;
            font-size: 15px;
            padding: 0 20px;
            transition: all 0.2s ease;
            border: none;
        }
        
        .stTabs [aria-selected="true"] {
            background: white;
            color: #1e293b;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            font-weight: 600;
        }
        
        /* Upload section styling */
        .upload-zone {
            border: 2px dashed #3b82f6;
            border-radius: 16px;
            padding: 3rem 2rem;
            text-align: center;
            background: rgba(55, 65, 81, 0.7);
            transition: all 0.3s ease;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            width: 100%;
            box-sizing: border-box;
        }
        
        .upload-zone:hover {
            border-color: #1d4ed8;
            background: rgba(55, 65, 81, 0.9);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
        }
        
        .upload-zone h4 {
            color: #ffffff !important;
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .upload-zone p {
            color: #e2e8f0 !important;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
        }
        
        /* File uploader styling */
        .stFileUploader {
            margin: 1rem 0;
            width: 100%;
            box-sizing: border-box;
        }
        
        .stFileUploader > div {
            width: 100%;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
        }
        
        .stFileUploader > div > div {
            background: rgba(55, 65, 81, 0.7);
            border: 1px solid #4b5563;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            text-align: center;
            width: 100%;
            box-sizing: border-box;
            margin: 0 auto;
        }
        
        /* Specific styling for file uploader button */
        .stFileUploader > div > div > div {
            width: 100% !important;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .stFileUploader > div > div > div > button {
            background: #374151 !important;
            border: 1px solid #4b5563 !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
            width: auto !important;
            min-width: 120px !important;
        }
        
        .stFileUploader > div > div > div > button:hover {
            background: #4b5563 !important;
            border-color: #6b7280 !important;
            transform: translateY(-1px) !important;
        }
        
        .stFileUploader label {
            color: #ffffff !important;
            font-weight: 500;
            margin-bottom: 0.5rem !important;
        }
        
        .stFileUploader small {
            color: #e2e8f0 !important;
            font-size: 0.85rem !important;
            margin-top: 0.5rem !important;
        }
        
        /* Ensure file uploader matches upload zone width */
        .upload-zone + * .stFileUploader {
            width: 100%;
            max-width: none;
        }
        
        /* Force consistent width for file uploader containers */
        div[data-testid="stFileUploader"] {
            width: 100% !important;
        }
        
        div[data-testid="stFileUploader"] > div {
            width: 100% !important;
        }
        
        /* Override any column constraints on file uploader */
        .stColumn .stFileUploader {
            width: 100% !important;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.3);
        }
        
        /* Card styling */
        .detection-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
            width: 100%;
            box-sizing: border-box;
        }
        
        /* Status badges */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 500;
            margin: 0.25rem 0.25rem 0.25rem 0;
        }
        
        .badge-mask { 
            background: #dcfce7; 
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        .badge-no-mask { 
            background: #fef2f2; 
            color: #dc2626;
            border: 1px solid #fecaca;
        }
        .badge-incorrect { 
            background: #dbeafe; 
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #ffffff;
            margin: 2rem 0 1rem 0;
            padding: 1rem 1.5rem;
            background: rgba(55, 65, 81, 0.8);
            border-radius: 12px;
            border-left: 4px solid #3b82f6;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            width: 100%;
            box-sizing: border-box;
        }
        
        /* Info boxes */
        .info-box {
            background: rgba(55, 65, 81, 0.7);
            border: 1px solid #4b5563;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            border-left: 4px solid #3b82f6;
            width: 100%;
            box-sizing: border-box;
        }
        
        .info-box h5 {
            color: #ffffff !important;
            font-weight: 600;
            margin-bottom: 0.75rem;
            font-size: 1.1rem;
        }
        
        .info-box p, .info-box li {
            color: #e2e8f0 !important;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        .info-box ul, .info-box ol {
            padding-left: 1.5rem;
        }
        
        /* Clear button styling */
        div[data-testid="stButton"] button[key="clear_image"],
        div[data-testid="stButton"] button:has-text("Clear Image") {
            background: #ef4444 !important;
            color: white !important;
            border: 1px solid #dc2626 !important;
            box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2) !important;
        }
        
        div[data-testid="stButton"] button[key="clear_image"]:hover,
        div[data-testid="stButton"] button:has-text("Clear Image"):hover {
            background: #dc2626 !important;
            border-color: #b91c1c !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(239, 68, 68, 0.3) !important;
        }
        
        /* Legacy clear button class */
        .clear-button {
            background: #ef4444 !important;
            color: white !important;
            border: 1px solid #dc2626 !important;
            box-shadow: 0 2px 4px rgba(239, 68, 68, 0.2) !important;
        }
        
        .clear-button:hover {
            background: #dc2626 !important;
            color: white !important;
            border-color: #b91c1c !important;
            transform: translateY(-1px) !important;
        }
        
        /* Progress bar styling - Clean blue fill */
        .stProgress > div > div {
            background-color: #1976D2 !important;
            border-radius: 8px !important;
            position: relative !important;
            overflow: hidden !important;
            transition: width 0.3s ease-in-out !important;
        }
        
        /* Add shimmer effect to active progress */
        .stProgress > div > div::after {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent) !important;
            animation: shimmer 2s infinite !important;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Video container styling */
        .stVideo {
            width: 100% !important;
            box-sizing: border-box;
        }
        
        .stVideo > div {
            width: 100% !important;
        }
        
        /* Image container styling */
        .stImage {
            width: 100% !important;
            box-sizing: border-box;
        }
        
        /* Column styling for consistent width */
        .stColumn {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        
        .stColumn:first-child {
            padding-left: 0;
        }
        
        .stColumn:last-child {
            padding-right: 0;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 100%;
            }
            
            .upload-zone {
                padding: 2rem 1rem;
                margin: 0.5rem 0;
            }
            
            .section-header {
                padding: 0.75rem 1rem;
                font-size: 1.25rem;
            }
            
            .info-box {
                padding: 1rem;
                margin: 0.5rem 0;
            }
            
            .detection-card {
                padding: 1rem;
                margin: 0.5rem 0;
            }
            
            .stFileUploader > div > div {
                padding: 0.75rem 1rem;
            }
            
            .stFileUploader > div > div > div > button {
                padding: 0.5rem 1rem !important;
                min-width: 100px !important;
            }
        }
        
        @media (max-width: 480px) {
            .main .block-container {
                padding-left: 0.75rem;
                padding-right: 0.75rem;
            }
            
            .upload-zone {
                padding: 1.5rem 0.75rem;
            }
            
            .upload-zone h4 {
                font-size: 1.1rem;
            }
            
            .upload-zone p {
                font-size: 0.85rem;
            }
            
            .stFileUploader > div > div {
                padding: 0.5rem 0.75rem;
            }
            
            .stFileUploader small {
                font-size: 0.8rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>Face Mask Detection Using YOLOv5</h1>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4>AI-Powered Face Mask Detection</h4>
        <p>Built with <strong>YOLOv5</strong> • <strong>Streamlit</strong> • <strong>OpenCV</strong></p>
        <br>
        <small>Real-time Detection • Smooth Tracking • High Accuracy</small>
    </div>
    """, unsafe_allow_html=True)


def render_upload_section(title: str, subtitle: str, supported_formats: str):
    """Render a simple upload section"""
    st.markdown(f"**{title}**")
    st.markdown(f"{subtitle}")
    st.markdown(f"*{supported_formats}*")


def render_feature_card(content: str, style_override: str = ""):
    """Render a simple content section"""
    st.markdown(content, unsafe_allow_html=True)


def render_detection_badge(class_name: str, confidence: float, icon: str, badge_class: str):
    """Render a simple detection result"""
    st.markdown(f"{icon} **{class_name}** - Confidence: {confidence:.1%}")


def get_detection_display_info(class_name: str):
    """Get display information for a detection class"""
    class_lower = class_name.lower()
    
    if 'without_mask' in class_lower or 'no_mask' in class_lower:
        return "With Mask", "badge-mask", ""
    elif 'with_mask' in class_lower:
        return "Without Mask", "badge-no-mask", ""
    else:
        return "Incorrect Mask", "badge-incorrect", ""
