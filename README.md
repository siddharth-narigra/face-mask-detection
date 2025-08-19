# 😷 Face Mask Detection

A modern, AI-powered web application for real-time face mask detection using YOLOv5 and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![YOLOv5](https://img.shields.io/badge/YOLOv5-7.0%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange)

## ✨ Features

- **🎯 Real-time Detection**: Advanced AI-powered mask detection with YOLOv5
- **🎥 Video Processing**: Process video files with smooth bounding box tracking
- **📸 Image Analysis**: Upload and analyze images for mask compliance
- **📹 Webcam Support**: Real-time detection from webcam (local setup)
- **🔄 Advanced Smoothing**: Exponential Moving Average (EMA) for jitter-free tracking
- **👥 Object Tracking**: Persistent ID tracking across video frames
- **🎨 Modern UI**: Professional interface with animated tabs and progress bars
- **📊 Color-coded Results**: Intuitive visual feedback for different mask states

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face-mask-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your model weights**
   - Add your trained YOLOv5 model (`best.pt`) to the `weights/` directory
   - Or download a pre-trained model

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   - Navigate to `http://localhost:8501`
   - Start detecting masks!

## 📁 Project Structure

```
face-mask-detection/
├── 📄 app.py                # Main entry point (minimal, imports others)
├── 📄 requirements.txt      # Dependencies
├── 📄 README.md             # This file
│
├── 📁 weights/              # Model weights
│   └── best.pt              # Your trained YOLOv5 model
│
├── 📁 src/                  # All source code lives here
│   ├── __init__.py
│   ├── 🤖 models.py         # Model loading & setup
│   ├── 🎥 video_processor.py # Video processing & smoothing
│   ├── 🖼️ image_processor.py # Image processing logic
│   ├── 📹 webcam_processor.py # Webcam functionality
│   └── 🎨 ui_components.py  # UI, styling, helpers
│
├── 📁 tests/                # Unit & integration tests
│   ├── test_models.py
│   ├── test_video_processor.py
│   └── __init__.py
│
├── 📁 utils/                # Helper modules
│   ├── logger.py            # Logging setup
│   ├── config.py            # Configuration settings
│   └── __init__.py
│
└── 📁 .venv/                # Virtual environment (ignored in git)
```

## 🎛️ Configuration

Key settings can be modified in `utils/config.py`:

- **Model Settings**: Confidence threshold, model path
- **Video Processing**: Smoothing algorithm, EMA alpha, IoU threshold
- **UI Settings**: Colors, display names, app title
- **Performance**: Codec preferences, buffer sizes

## 🔧 Advanced Features

### Smoothing Algorithms
- **EMA (Exponential Moving Average)**: Best for real-time, responsive smoothing
- **Moving Average**: Traditional averaging over N frames
- **Hybrid**: Combines EMA + moving average for ultra-smooth results

### Object Tracking
- IoU-based detection matching
- Persistent track IDs across frames
- Automatic cleanup of disappeared objects
- Handles occlusions and temporary disappearances

### Video Codec Support
- Automatic codec fallback (H264 → XVID → MJPG → mp4v)
- Browser-compatible output
- Optimized for streaming and download

## 🧪 Testing

Run the test suite:
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v tests/
```

## 📊 Detection Classes

The model detects three mask-wearing states:

- **✅ With Mask** (Green): Person properly wearing a mask
- **❌ Without Mask** (Red): Person not wearing a mask
- **⚠️ Incorrect Mask** (Blue): Person wearing mask incorrectly

## 🎨 UI Components

- **Modern Tabs**: Smooth animated tab system
- **Progress Tracking**: Real-time processing progress
- **Drag & Drop**: Easy file upload interface
- **Responsive Design**: Works on desktop and mobile
- **Professional Styling**: Gradient headers, hover effects, shadows

## 🛠️ Development

### Adding New Features
1. Create new module in `src/`
2. Add configuration to `utils/config.py`
3. Write tests in `tests/`
4. Update imports in `src/__init__.py`
5. Update this README

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings to functions
- Keep functions focused and modular

## 📋 Requirements

See `requirements.txt` for full dependency list. Key packages:

- **streamlit**: Web app framework
- **torch**: PyTorch for model inference
- **opencv-python**: Computer vision operations
- **yolov5**: Object detection model
- **pillow**: Image processing
- **numpy**: Numerical operations

## 🐛 Troubleshooting

### Common Issues

1. **Model fails to load**
   - Check that `weights/best.pt` exists
   - Verify model file isn't corrupted
   - Ensure sufficient disk space

2. **Video processing slow**
   - Reduce video resolution
   - Adjust EMA alpha in config
   - Check available system resources

3. **Webcam not working**
   - Ensure camera permissions are granted
   - Run app locally (not on cloud platforms)
   - Check camera isn't used by other applications

### Performance Tips

- Use GPU if available (CUDA-enabled PyTorch)
- Process smaller video files for faster results
- Adjust confidence threshold for speed vs accuracy trade-off
- Use EMA smoothing for best performance

## 📈 Future Enhancements

- [ ] Real-time webcam streaming in browser
- [ ] Batch processing for multiple files
- [ ] Detection analytics and reporting
- [ ] Custom model training interface
- [ ] Mobile app version
- [ ] Cloud deployment guides

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv5** by Ultralytics for the detection model
- **Streamlit** team for the amazing web app framework
- **OpenCV** community for computer vision tools
- Contributors and testers who helped improve this project

---

**Built with ❤️ for public health and safety**
