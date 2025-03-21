from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import cv2
import torch
import numpy as np
from utils.kernel_utils import VideoReader, FaceExtractor
from utils.model import DeepFakeClassifier
import base64



app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variable for models
MODELS = []

# Load models at startup
def load_models():
    global MODELS
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    
    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)
    
    # Print all files in weights directory
    print("Available weight files:")
    for file in os.listdir(weights_dir):
        print(f"- {file}")
        
    # Load all model files that match the pattern
    model_paths = [
        os.path.join(weights_dir, f) 
        for f in os.listdir(weights_dir) 
        if 'DeepFakeClassifier' in f
    ]
    
    print(f"\nLoading {len(model_paths)} models:")
    for path in model_paths:
        try:
            model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
            model.eval()
            MODELS.append(model.half())
            print(f"✓ Loaded model: {path}")
        except Exception as e:
            print(f"✗ Error loading model {path}: {str(e)}")
    
    print(f"\nSuccessfully loaded {len(MODELS)} models")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_with_models(face_tensor):
    predictions = []
    with torch.no_grad():
        for i, model in enumerate(MODELS):
            try:
                # Ensure input is in half precision
                face_tensor = face_tensor.half()
                pred = model(face_tensor)
                pred = torch.sigmoid(pred).cpu().numpy()[0][0]
                predictions.append(pred)
                print(f"Model {i+1} prediction: {pred:.4f}")
            except Exception as e:
                print(f"Error in model {i+1}: {str(e)}")
                continue
    
    if predictions:
        avg_pred = np.mean(predictions)
        print(f"Average prediction: {avg_pred:.4f} ({'FAKE' if avg_pred > 0.5 else 'REAL'})")
        return avg_pred
    return 0.0

def process_video(video_path):
    try:
        print("\n=== Starting Video Analysis ===")
        
        video_reader = VideoReader()
        video_read_fn = lambda x: video_reader.read_frames(x, num_frames=32)
        face_extractor = FaceExtractor(video_read_fn)
        
        frames_data = face_extractor.process_video(video_path)
        print(f"Found {len(frames_data)} frames with faces")
        
        faces_data = []
        
        for frame_idx, frame_data in enumerate(frames_data):
            if 'faces' not in frame_data or not frame_data['faces']:
                continue
                
            for face_idx, face in enumerate(frame_data['faces']):
                if face is None or not isinstance(face, np.ndarray):
                    continue
                
                try:
                    face = face.astype(np.uint8)
                    if face.size == 0:
                        continue
                        
                    # Convert BGR to RGB
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    
                    # Prepare face for model
                    face_tensor = prepare_face_for_model(face_rgb)
                    
                    # Get prediction
                    prediction = float(predict_with_models(face_tensor))  # Convert to Python float
                    
                    # Prepare face for display
                    display_face = cv2.resize(face_rgb, (300, 300))
                    _, buffer = cv2.imencode('.jpg', display_face)
                    face_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Store face data with its prediction
                    faces_data.append({
                        'face': face_base64,
                        'prediction': prediction,
                        'is_fake': bool(prediction > 0.5)  # Convert to Python bool
                    })
                    
                except Exception as e:
                    print(f"Error processing face: {str(e)}")
                    continue
        
        # Calculate overall prediction
        if faces_data:
            overall_pred = float(np.mean([face['prediction'] for face in faces_data]))  # Convert to Python float
        else:
            overall_pred = 0.0
            
        print("\n=== Final Results ===")
        print(f"Number of faces processed: {len(faces_data)}")
        print(f"Overall prediction: {overall_pred:.4f}")
        for idx, face_data in enumerate(faces_data):
            print(f"Face {idx+1}: {face_data['prediction']:.4f} ({'FAKE' if face_data['is_fake'] else 'REAL'})")
        print("==================\n")
        
        return {
            'success': True,
            'overall_prediction': overall_pred,
            'faces_data': faces_data,
            'message': f'Processed {len(faces_data)} faces successfully'
        }
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {
            'success': False,
            'message': f'Error processing video: {str(e)}'
        }

def prepare_face_for_model(face):
    # Resize and normalize face image for model input
    face = cv2.resize(face, (380, 380))
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face_tensor = torch.from_numpy(face).unsqueeze(0).cuda()
    # Convert to half precision to match model
    face_tensor = face_tensor.half()
    return face_tensor

# Add a cleanup function to remove old files
def cleanup_uploads():
    """Remove all files from uploads directory"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {str(e)}')

# Clean up uploads folder when starting the app
cleanup_uploads()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file uploaded'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No video selected'}), 400

        # Create uploads directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the uploaded video with a secure filename
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(f"Saving video to: {video_path}")
        video_file.save(video_path)

        # Process the video
        result = process_video(video_path)
        
        # Clean up
        try:
            os.remove(video_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file: {str(e)}")

        return jsonify(result)

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    load_models()  # Load models before starting the app
    app.run(debug=True) 