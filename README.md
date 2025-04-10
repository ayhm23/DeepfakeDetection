# Deepfake Detection Web Application

This web application provides a user-friendly interface for detecting deepfake videos using deep learning models. It analyzes uploaded videos frame by frame, detecting faces and determining whether they are real or artificially manipulated (deepfakes).

## Features

- Web-based interface for video upload
- Real-time deepfake detection
- Face extraction and analysis
- Multiple model ensemble for improved accuracy
- Visual feedback with confidence scores
- Support for various video formats

## Technical Architecture

- **Frontend**: HTML/CSS/JavaScript
- **Backend**: Flask (Python)
- **Deep Learning**: PyTorch
- **Face Detection**: OpenCV
- **Model**: EfficientNet-B7 based classifier

## Project Structure

```
deepfake-detection-web/
├── app.py              # Main Flask application
├── templates/          # HTML templates
├── static/            # Static files (CSS, JS)
├── uploads/           # Temporary video upload directory
├── weights/           # Model weight files
├── utils/             # Utility functions
│   ├── kernel_utils.py
│   └── model.py
└── background image/  # UI assets
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ayhm23/DeepfakeDetection.git
cd deepfake-detection-web
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv        # use python3.10
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model weights:
https://1drv.ms/f/c/363bbf1f20a036e0/EqGZ9V7lFShHq6as6BE-DpEBuEQlHPKBC60Njc5riYWuoA?e=lVADuK
Place the DeepFakeClassifier model weights in the `weights/` directory.

## Usage

1. Start the server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

3. Upload a video through the web interface and wait for the analysis results.

## Model Information

The application uses an ensemble of EfficientNet-B7 based models for deepfake detection. The models analyze facial features and patterns to determine if a video has been manipulated. The prediction is made by averaging the predictions from multiple models for improved reliability.

### Model Architecture Details:
- **Base Model**: EfficientNet-B7 (Noisy Student)
- **Input Size**: 380x380 pixels
- **Precision**: Half-precision (FP16) for faster inference
- **Ensemble Method**: Weighted average of multiple model predictions
- **Face Processing**: 
  - Face detection and extraction using OpenCV
  - RGB conversion and normalization
  - Tensor preparation for GPU inference

### Model Features:
- Multi-frame analysis for temporal consistency
- Face-specific feature extraction
- Confidence scoring for each detected face
- Ensemble prediction for improved reliability

## About Me

### Team Members
1. Archit Jaju [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/archit-jaju1/)
2. Sanyam Verma  [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sanyam-verma1231/)
3. Sarthak Raj [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sarthak-raj-6588b2309/)
4. Kunal Narang [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kunal-narang-76816b18a/)

## Technical Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Minimum 8GB RAM
- Sufficient storage space for model weights

## Dependencies

See `requirements.txt` for a complete list of Python dependencies.

## Security Notes

- The application includes automatic cleanup of uploaded files
- Secure filename handling for uploads
- Temporary file storage management

## Limitations

- Processing time depends on video length and hardware capabilities
- Requires clear facial visibility for accurate detection
- May have reduced accuracy on heavily compressed videos

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here] 
