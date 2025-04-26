import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import uuid
import base64
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from models import AudioDetectionResult, DetectionHistory, ImageDetectionResult, PublicDetections, User, UserFavourties, VideoDetectionResult, db
from config import Config
import numpy as np
import cv2
from PIL import Image, ImageChops, ExifTags 
import librosa
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from skimage.util import random_noise
from fractions import Fraction
from PIL.ExifTags import TAGS
import json
import time
import subprocess
import matplotlib
matplotlib.use('Agg')  # Prevents Tkinter from being used
import scipy.signal 
from scipy import signal

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Load ML models
image_model_path = 'D:/flask_server_fyp/ml_modals/custom_augmented_model.h5'
audio_model_path = 'D:/flask_server_fyp/ml_modals/deep_fake_audio_detector.h5'
video_model_path = 'D:/flask_server_fyp/ml_modals/deepfake_detection_model.h5'

# Load models
image_model = tf.keras.models.load_model(image_model_path)
audio_model = tf.keras.models.load_model(audio_model_path)
video_model = tf.keras.models.load_model(video_model_path)

# Load feature extractor for video processing
base_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')

# Configure detection upload folder
BASE_DIR = os.getcwd()  # Get the absolute path of the current working directory
DETECTION_UPLOAD_FOLDER = os.path.join(BASE_DIR, "detection_uploads")
if not os.path.exists(DETECTION_UPLOAD_FOLDER):
    os.makedirs(DETECTION_UPLOAD_FOLDER)
app.config['DETECTION_UPLOAD_FOLDER'] = DETECTION_UPLOAD_FOLDER

DETECTION_VIDEO_FRAME_UPLOAD_FOLDER = os.path.join(BASE_DIR, "detection_uploads/video_frames")
if not os.path.exists(DETECTION_VIDEO_FRAME_UPLOAD_FOLDER):
    os.makedirs(DETECTION_VIDEO_FRAME_UPLOAD_FOLDER)
app.config['DETECTION_VIDEO_FRAME_UPLOAD_FOLDER'] = DETECTION_VIDEO_FRAME_UPLOAD_FOLDER

# Ensure the profile picture directory exists
UPLOAD_FOLDER = os.path.join(BASE_DIR, "profile_pics")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize database
db.init_app(app)

# Enable CORS for all routes
CORS(app)

# Create database tables
with app.app_context():
    db.create_all()

# Helper function for error handling
def handle_error(message, status_code):
    return jsonify({"error": message, "success": False}), status_code

# Register a new user
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.form
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        profile_picture = request.files.get('profile_picture')

        if not all([username, password, email, profile_picture]):
            return handle_error("All fields are required", 400)

        # Check if username already exists
        if User.query.filter_by(username=username).first():
            return handle_error("Username already exists", 400)

        # Secure the filename and add timestamp to avoid overwriting
        filename = secure_filename(profile_picture.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)

        profile_picture.save(save_path)  # Save the profile picture

        # Verify file is actually saved
        if not os.path.exists(save_path):
            return handle_error("File upload failed", 500)

        # Hash password before saving
        hashed_password = generate_password_hash(password)

        # Store user in the database with image filename
        new_user = User(
            username=username,
            password=password,
            email=email,
            profile_picture=saved_filename  # Only save the filename in the database
        )
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "user": {
                "id": new_user.id,
                "username": new_user.username,
                "email": new_user.email,
                "profile_picture": new_user.profile_picture
            }
        }), 201 

    except Exception as e:
        return handle_error(str(e), 500)

# Login route
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data.get('email')
        login_password = data.get('password')

        if not email or not login_password:
            return handle_error("Email and password are required", 400)

        user = User.query.filter_by(email=email).first()
        if not user:
            return handle_error("User not found", 404)
        # user_pass = generate_password_hash(login_password) 
        # Check password
        if not user.check_password(login_password):  # ✅ Use the method from the User model
            return handle_error("Invalid credentials", 401)


        # Generate image URL
        profile_picture_url = f"/images/{user.profile_picture}" if user.profile_picture else None

        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "profile_picture": profile_picture_url 
            }
        }), 200

    except Exception as e:
        return handle_error(str(e), 500)

# Get all users
@app.route('/users', methods=['GET'])
def get_users():
    try:
        users = User.query.all()
        user_list = [{
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "profile_picture": f"/images/{user.profile_picture}" if user.profile_picture else None
        } for user in users]

        return jsonify(user_list), 200
    except Exception as e:
        return handle_error(str(e), 500)

# Get a specific user by ID
@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return handle_error("User not found", 404)

        return jsonify({
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "profile_picture": f"images/{user.profile_picture.lstrip('/')}" if user.profile_picture else ""
        }), 200
    except Exception as e:
        return handle_error(str(e), 500)

# Serve profile images
@app.route("/images/<filename>")
def get_image(filename):
    try:
        print(f"Serving image: {filename} from {UPLOAD_FOLDER}")  # Debugging log
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return handle_error("Image not found", 404) 
    
@app.route('/update-profile', methods=['POST'])
def update_profile():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        user_id = data.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        user = User.query.filter_by(id=user_id).first()
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Only update fields that are provided and not empty
        updated = False
        
        if 'username' in data and data['username'].strip():
            user.username = data['username'].strip()
            updated = True
            
        if 'email' in data and data['email'].strip():
            user.email = data['email'].strip()
            updated = True
            
        if 'password' in data and data['password'].strip():
            user.password = generate_password_hash(data['password'].strip())
            updated = True

        if not updated:
            return jsonify({"error": "No valid fields provided for update"}), 400

        db.session.commit()

        return jsonify({
            "message": "Profile updated successfully",
            "username": user.username,
            "email": user.email
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
        
@app.route('/update-profile-picture', methods=['POST'])
def update_profile_picture():
    try:
        # Check if request has the file part
        if 'profile_picture' not in request.files:
            return handle_error("No file part in request", 400)
            
        file = request.files['profile_picture']
        
        # Check if file is empty
        if file.filename == '':
            return handle_error("No selected file", 400)
            
        # Check if file is an image
        if not allowed_file(file.filename):
            return handle_error("Invalid file type", 400)
            
        # Get user ID from form data
        user_id = request.form.get('user_id')
        if not user_id:
            return handle_error("User ID is required", 400)
            
        try:
            user_id = int(user_id)
        except ValueError:
            return handle_error("Invalid user ID", 400)
            
        # Find user in database
        user = User.query.get(user_id)
        if not user:
            return handle_error("User not found", 404)
            
        # Generate new filename
        file_ext = os.path.splitext(file.filename)[1].lower()
        new_filename = f"profile_{user_id}{file_ext}"
        new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        
        # Remove old image if exists
        if user.profile_picture:
            old_filepath = os.path.join(app.config['UPLOAD_FOLDER'], user.profile_picture)
            if os.path.exists(old_filepath):
                os.remove(old_filepath)
                
        # Save new image
        file.save(new_filepath)
        
        # Update database
        user.profile_picture = new_filename
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": "Profile picture updated successfully",
            "profile_picture": new_filename
        }), 200
        
    except Exception as e:
        app.logger.error(f"Error updating profile picture: {str(e)}")
        return handle_error("Internal server error", 500)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/results/user/<int:user_id>', methods=['GET'])
def get_all_results(user_id):
    """Get all detection results for a user."""
    try:
        results = DetectionHistory.query.filter_by(user_id=user_id).all()
        if not results:
            return handle_error("No detection history found for this user.", 404)

        response = [{
            "detection_id": result.detection_id,
            "detection_type": result.detection_type,
            "timestamp": result.created_at
        } for result in results]

        return jsonify(response)
    
    except Exception as e:
        return handle_error(f"Error fetching user results: {str(e)}", 500)

@app.route('/results/audio/<int:detection_id>', methods=['GET'])
def get_audio_result(detection_id):
    """Get audio detection result by detection ID."""
    try:
        result = AudioDetectionResult.query.get(detection_id)
        if not result:
            return handle_error("Audio detection result not found.", 404)

        response = {
            "detection_id": result.id,
            "user_id": result.user_id,
            "filename": result.filename,
            "label": result.label,
            "confidence": result.confidence,
            "audio_metadata": result.audio_metadata,
            "spectrogram_path": result.spectrogram_path,
            "frequency_graph_path": result.frequency_graph_path,
            "frequencies": result.frequencies,  
            "noise_level": result.noise_level,  
            "echo_peak": result.echo_peak,   
            'isFavorite': result.favorite,
            'public': result.public,
            "timestamp": result.created_at
        }

        return jsonify(response)

    except Exception as e:
        return handle_error(f"Error fetching audio result: {str(e)}", 500)
    
@app.route('/results/image/<int:detection_id>', methods=['GET'])
def get_image_result(detection_id):
    """Get image detection result by detection ID."""
    try:
        result = ImageDetectionResult.query.get(detection_id)
        if not result:
            return handle_error("Image detection result not found.", 404)

        response = {
            "detection_id": result.id,
            "user_id": result.user_id,
            "filename": result.filename,
            "label": result.label,
            "confidence": result.confidence,
            "image_metadata": result.image_metadata,
            "ela_image": result.ela_image,
            "jpeg_image": result.jpeg_image,
            "noise_image": result.noise_image,
            "heat_image": result.heatmap_image,
            'isFavorite': result.favorite,
            'public': result.public,
            "timestamp": result.created_at
        }

        return jsonify(response)

    except Exception as e:
        return handle_error(f"Error fetching image result: {str(e)}", 500)


@app.route('/results/video/<int:detection_id>', methods=['GET'])
def get_video_result(detection_id):
    """Get video detection result by detection ID."""
    try:
        result = VideoDetectionResult.query.get(detection_id)
        if not result:
            return handle_error("Video detection result not found.", 404)

        response = {
            "detection_id": result.id,
            "user_id": result.user_id,
            "filename": result.filename,
            "label": result.label,
            "confidence": result.confidence,
            "video_metadata": result.video_metadata,
            "frame_analysis_image": result.frame_analysis_image,
            'isFavorite': result.favorite,
            'public': result.public,
            "timestamp": result.created_at
        }

        return jsonify(response)

    except Exception as e:
        return handle_error(f"Error fetching video result: {str(e)}", 500)
    
@app.route("/detection-files/<filename>")
def get_detection_files(filename):
    try:
        print(f"Serving image: {filename} from {DETECTION_UPLOAD_FOLDER}")  # Debugging log
        return send_from_directory(app.config['DETECTION_UPLOAD_FOLDER'], filename)
    except Exception as e:
        return handle_error("Image not found", 404)
    
@app.route("/video-frame-files/<filename>")
def get_video_detection_files(filename):
    try:
        print(f"Serving image: {filename} from {DETECTION_VIDEO_FRAME_UPLOAD_FOLDER}")  # Debugging log
        return send_from_directory(app.config['DETECTION_VIDEO_FRAME_UPLOAD_FOLDER'], filename)
    except Exception as e:
        return handle_error("Image not found", 404)

@app.route('/toggle_favorite', methods=['POST'])
def toggle_favorite():
    data = request.get_json()
    detection_type = data.get('detection_type')
    detection_id = data.get('detection_id')

    if not detection_type or not detection_id:
        return jsonify({'error': 'Missing detection_type or detection_id'}), 400

    # Determine which model to use based on detection type
    if detection_type == 'image':
        model = ImageDetectionResult
    elif detection_type == 'audio':
        model = AudioDetectionResult
    elif detection_type == 'video':
        model = VideoDetectionResult
    else:
        return jsonify({'error': 'Invalid detection type'}), 400

    # Find the record
    record = model.query.get(detection_id)
    if not record:
        return jsonify({'error': 'Record not found'}), 404

    # Toggle the favorite status
    record.favorite = not record.favorite
    db.session.commit()  # Save the change

    # Get user_id after updating the record
    user_id = record.user_id  # Ensure user_id exists
    favorite_value = record.favorite  # Get the updated favorite status

    if favorite_value:  # If favorited, add to UserFavourites
        new_fav = UserFavourties(
            user_id=user_id,
            detection_type=detection_type,
            detection_id=detection_id,
        )
        db.session.add(new_fav)
    else:  # If unfavorited, remove the entry
        UserFavourties.query.filter_by(
            user_id=user_id, detection_type=detection_type, detection_id=detection_id
        ).delete()

    db.session.commit()  # Commit changes

    return jsonify({
        'success': True,
        'favorite': favorite_value
    })

@app.route('/results/userFavorites/<int:user_id>', methods=['GET'])
def get_user_favorites(user_id):
    try:
        # Query all favorites for the user
        favorites = UserFavourties.query.filter_by(user_id=user_id).all()
        
        if not favorites:
            return jsonify([])  # Return empty array if no favorites
        
        favorite_detections = []
        
        for fav in favorites:
            # Determine which model to query based on detection type
            if fav.detection_type == 'image':
                model = ImageDetectionResult
            elif fav.detection_type == 'audio':
                model = AudioDetectionResult
            elif fav.detection_type == 'video':
                model = VideoDetectionResult
            else:
                continue  # Skip invalid types
            
            # Get the detection record
            detection = model.query.get(fav.detection_id)
            if detection:
                # Format the response similarly to your existing history endpoint
                favorite_detections.append({
                    'detection_id': detection.id,
                    'detection_type': fav.detection_type,
                    'user_id': user_id,
                    'created_at': detection.created_at.isoformat() if detection.created_at else None,
                    # Include any other fields your frontend expects
                })
        
        return jsonify(favorite_detections)
    
    except Exception as e:
        app.logger.error(f"Error fetching favorites for user {user_id}: {str(e)}")
        return jsonify({'error': 'Failed to fetch favorites'}), 500

@app.route('/allow_public', methods=['POST'])
def allow_public():
    data = request.get_json()
    detection_type = data.get('detection_type')
    detection_id = data.get('detection_id')

    if not detection_type or not detection_id:
        return jsonify({'error': 'Missing detection_type or detection_id'}), 400

    # Determine which model to use based on detection type
    if detection_type == 'image':
        model = ImageDetectionResult
    elif detection_type == 'audio':
        model = AudioDetectionResult
    elif detection_type == 'video':
        model = VideoDetectionResult
    else:
        return jsonify({'error': 'Invalid detection type'}), 400

    # Find the record
    record = model.query.get(detection_id)
    if not record:
        print("Record not found")
        return jsonify({'error': 'Record not found'}), 404

    # Toggle the public status
    record.public = not record.public
    db.session.commit()  # Save the change

    # Get user_id after updating the record
    user_id = record.user_id

    if record.public:  # If made public, add to PublicDetections
        new_public = PublicDetections(
            user_id=user_id,
            detection_type=detection_type,
            detection_id=detection_id,
        )
        db.session.add(new_public)
    else:  # If made private, remove the entry
        PublicDetections.query.filter_by(
            user_id=user_id, detection_type=detection_type, detection_id=detection_id
        ).delete()

    db.session.commit()  # Commit changes

    return jsonify({
        'success': True,
        'public': record.public
    })
    
@app.route('/results/public_detections', methods=['GET'])
def get_public_detections():
    try:
        # Query all public detections
        public_entries = PublicDetections.query.order_by(PublicDetections.created_at.desc()).all()
        
        if not public_entries:
            return jsonify([])  # Return empty array if no public detections
        
        public_detections = []
        
        for entry in public_entries:
            # Determine which model to query based on detection type
            if entry.detection_type == 'image':
                model = ImageDetectionResult
            elif entry.detection_type == 'audio':
                model = AudioDetectionResult
            elif entry.detection_type == 'video':
                model = VideoDetectionResult
            else:
                continue  # Skip invalid types
            
            # Get the detection record
            detection = db.session.get(model, entry.detection_id)
            if detection and detection.public:  # Double-check the public flag
                # Format the response with common fields
                public_detections.append({
                    'detection_id': detection.id,
                    'detection_type': entry.detection_type,
                    'user_id': entry.user_id,
                    'created_at': entry.created_at.isoformat(),
                })
        
        return jsonify(public_detections)
    
    except Exception as e:
        app.logger.error(f"Error fetching public detections: {str(e)}")
        return jsonify({'error': 'Failed to fetch public detections'}), 500
# ======================
# DEEPFAKE DETECTION ROUTES
# ======================  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from skimage.util import random_noise
import cv2

def get_image_metadata(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    
    metadata = {}
    if exif_data:
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            
            # Ensure proper JSON serialization
            if isinstance(value, bytes):
                value = value.decode(errors="ignore")  # Decode bytes safely
            elif isinstance(value, tuple):
                value = [float(v) if isinstance(v, (int, float)) else str(v) for v in value]
            elif isinstance(value, (int, float)):
                value = float(value)  # Convert numeric values
            elif isinstance(value, str):
                try:
                    value = float(value)  # Convert numeric strings to float
                except ValueError:
                    pass  # Keep non-numeric strings unchanged
            else:
                value = str(value)  # Convert other types to string

            metadata[tag_name] = value

    return metadata  # Return dictionary, NOT JSON string

def convert_ifd_rational(obj):
    if isinstance(obj, tuple):
        return tuple(float(x) if isinstance(x, Fraction) else str(x) for x in obj)
    elif isinstance(obj, Fraction):  # Handle single IFDRational values
        return float(obj)
    return obj

def error_level_analysis(image_path, output_path):
    original = Image.open(image_path).convert("RGB")

    # Save a recompressed version at lower quality
    recompressed_path = "recompressed_image.jpg"
    original.save(recompressed_path, "JPEG", quality=50)
    
    recompressed = Image.open(recompressed_path)
    
    # Compute difference and normalize
    diff = ImageChops.difference(original, recompressed).convert("L")
    diff_np = np.array(diff)
    diff_np = np.log1p(diff_np)  # Enhance visibility of differences

    plt.figure(figsize=(10, 8))
    plt.imshow(diff_np, cmap="hot")
    plt.colorbar(label="Error Intensity")  # Add color bar
    plt.title("Error Level Analysis (ELA)", fontsize=16, pad=20)

    # Add text annotations with better positioning
    plt.annotate("High intensity → Possible manipulations", xy=(10, 20), fontsize=12, color='white', backgroundcolor='black', bbox=dict(facecolor='black', alpha=0.8))
    plt.annotate("Low intensity → Original areas", xy=(10, 40), fontsize=12, color='white', backgroundcolor='black', bbox=dict(facecolor='black', alpha=0.8))

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
def jpeg_compression_analysis(image_path, output_path):
    original = Image.open(image_path).convert("RGB")
    original_np = np.array(original)

    # Save the image with low quality
    compressed_path = "compressed_image.jpg"
    original.save(compressed_path, "JPEG", quality=30)
    
    compressed = Image.open(compressed_path)
    compressed_np = np.array(compressed)

    # Compute absolute difference
    difference = np.abs(original_np - compressed_np)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image", fontsize=14, pad=10)
    
    plt.subplot(1, 3, 2)
    plt.imshow(compressed)
    plt.title("Compressed Image", fontsize=14, pad=10)

    plt.subplot(1, 3, 3)
    plt.imshow(difference, cmap="hot")
    plt.colorbar(label="Compression Artifacts")
    plt.title("Difference Heatmap", fontsize=14, pad=10)

    # Explanation text with better positioning
    plt.figtext(0.5, 0.01, "High compression artifacts indicate possible tampering", wrap=True, horizontalalignment='center', fontsize=12, color='white', backgroundcolor='black', bbox=dict(facecolor='black', alpha=0.8))

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
def noise_analysis(image_path, output_path):
    original = cv2.imread(image_path)
    
    # Generate noisy image
    noisy_image = random_noise(original, mode="s&p", amount=0.05)
    noisy_image = np.array(255 * noisy_image, dtype="uint8")

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image", fontsize=14, pad=10)

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    plt.title("Noisy Image", fontsize=14, pad=10)

    # Explanation with better positioning
    plt.figtext(0.5, 0.01, "Salt & Pepper noise simulates possible alterations", wrap=True, horizontalalignment='center', fontsize=12, color='white', backgroundcolor='black', bbox=dict(facecolor='black', alpha=0.8))

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def apply_heatmap(image_path, output_path):
    try:
        # Read the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to reduce noise and smooth intensity transitions
        blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate the intensity gradient using the Sobel operator (detects edges and intensity changes)
        grad_x = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred_gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the magnitude of the gradient (edge intensity)
        grad_magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalize the gradient magnitude to [0, 255] for visualization
        grad_magnitude = cv2.normalize(grad_magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a heatmap to the gradient magnitude (visualize intensity changes)
        heatmap = cv2.applyColorMap(np.uint8(grad_magnitude), cv2.COLORMAP_RAINBOW)

        # Superimpose the heatmap on the original image with transparency
        blended_heatmap = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

        # Save the output heatmap image
        cv2.imwrite(output_path, blended_heatmap)
        return True  # Success
    except Exception as e:
        print(f"Error in applying advanced heatmap: {str(e)}")
        return False  # Failure
@app.route('/detect/image', methods=['POST'])
def detect_image():
    if 'file' not in request.files or 'user_id' not in request.form:
        return handle_error("No file or user ID provided", 400)
    
    file = request.files['file']
    user_id = request.form['user_id']

    if file.filename == '':
        return handle_error("No selected file", 400)

    # Save the file permanently
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        # Get metadata
        metadata = get_image_metadata(save_path)
        metadata_str = json.dumps(metadata)
        
        # Perform ELA
        ela_filename = f'ela_{uuid.uuid4()}.png'
        ela_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], ela_filename)
        error_level_analysis(save_path, ela_path)

        # Perform JPEG compression analysis
        jpeg_filename = f'jpeg_{uuid.uuid4()}.png'
        jpeg_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], jpeg_filename)
        jpeg_compression_analysis(save_path, jpeg_path)

        # Perform noise analysis
        noise_filename = f'noise_{uuid.uuid4()}.png'
        noise_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], noise_filename)
        noise_analysis(save_path, noise_path)
        # Apply Advanced Heatmap (Lens Distortion Check)
        heatmap_filename = f'heatmap_{uuid.uuid4()}.png'
        heatmap_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], heatmap_filename)
        heatmap_success = apply_heatmap(save_path, heatmap_path)

        if not heatmap_success:
            return handle_error("Heatmap generation failed", 500)
        # Get model prediction
        image_label, image_confidence = check_fake_or_real(save_path)
        confidence_value = float(image_confidence)

        # Save the detection result to the database
        image_result = ImageDetectionResult(
            user_id=user_id,
            filename=filename,
            label=image_label,
            confidence=confidence_value,   
            image_metadata=metadata_str,
            ela_image=ela_filename,
            jpeg_image=jpeg_filename,
            noise_image=noise_filename, 
            heatmap_image=heatmap_filename, 
        )

        db.session.add(image_result)
        db.session.commit()
        
        # Save to the common Detection table
        detection_history = DetectionHistory(
            user_id=user_id,
            detection_type='image',
            detection_id=image_result.id
        )
        db.session.add(detection_history)
        db.session.commit()

        return jsonify({
            "id": image_result.id,
            "type": "image",
            "label": image_label,
            "confidence": confidence_value,
            "filename": filename,
            "metadata": metadata,
            "ela_image_url": ela_filename,
            "jpeg_image_url": jpeg_filename,
            "noise_image_url": noise_filename,
            "heatmap_image_url": heatmap_filename
        })
    except Exception as e:
        return handle_error(f"Error processing image: {str(e)}", 500)

# ==============Audio=================
@app.route('/detect/audio', methods=['POST'])
def detect_audio_route():
    if 'file' not in request.files or 'user_id' not in request.form:
        return handle_error("No file or user ID provided", 400)
    
    file = request.files['file']
    user_id = int(request.form['user_id'])  # Ensure integer conversion
 
    if file.filename == '':
        return handle_error("No selected file", 400)

    # Save the file permanently
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        # Extract metadata
        metadata = get_audio_metadata(save_path)

        # Generate file names for spectrogram and frequency graph
        spectrogram_filename = f'spectrogram_{filename}.png'
        frequency_graph_filename = f'frequency_{filename}.png'

        # Full paths for saving files
        spectrogram_full_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], spectrogram_filename)
        frequency_graph_full_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], frequency_graph_filename)

        # Perform spectral and acoustic analysis
        plot_spectrogram(save_path, spectrogram_full_path)
        frequencies = analyze_frequencies(save_path, frequency_graph_full_path)

        # Perform background noise consistency analysis
        noise_level = detect_background_noise(save_path)

        # Perform echo and reverberation analysis
        echo_peak = detect_echo(save_path)

        # Process the audio with the ML model
        label, confidence = detect_audio(save_path)

        # Save the detection result to the database
        audio_result = AudioDetectionResult(
            user_id=user_id,
            filename=filename,
            label=label,
            confidence=confidence,  # Already a float
            audio_metadata=json.dumps(metadata),  # Store metadata as JSON string
            spectrogram_path=spectrogram_filename,  # Store only the file name
            frequency_graph_path=frequency_graph_filename,  # Store only the file name
            frequencies=json.dumps(frequencies),  # Now frequencies is a list of native Python integers
            noise_level=noise_level,
            echo_peak=echo_peak
        )
        db.session.add(audio_result)
        db.session.commit()

        # Save to the common Detection table
        detection_history = DetectionHistory(
            user_id=user_id,
            detection_type='audio',
            detection_id=audio_result.id
        )
        db.session.add(detection_history)
        db.session.commit()

        return jsonify({
            "id": audio_result.id,
            "type": "audio",
            "label": label,
            "confidence": confidence,
            "filename": filename,
            "metadata": metadata,
            "spectrogram_url": spectrogram_filename,  # Return only the file name
            "frequency_graph_url": frequency_graph_filename,  # Return only the file name
            "frequencies": json.dumps(frequencies),  # Still return numerical values if needed
            "noise_level": noise_level,
            "echo_peak": echo_peak
        })

    except Exception as e:
        return handle_error(f"Error processing audio: {str(e)}", 500)
    
def get_audio_metadata(audio_path):
    result = subprocess.run(
        ["D:/flask_server_fyp/ml_modals/exiftool.exe", "-json", audio_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    
    if result.returncode == 0:
        metadata = json.loads(result.stdout)[0]
        # Ensure all values in metadata are JSON serializable
        for key, value in metadata.items():
            if isinstance(value, (np.integer, np.floating)):
                metadata[key] = int(value) if isinstance(value, np.integer) else float(value)
        return metadata
    else:
        raise Exception("Failed to extract metadata")

def plot_spectrogram(audio_path, output_path, fmin=20, fmax=8000):
    """
    Generate and save a professional-quality spectrogram
    
    Args:
        audio_path (str): Path to input audio file
        output_path (str): Path to save the spectrogram image
        fmin (int): Minimum frequency to display (Hz)
        fmax (int): Maximum frequency to display (Hz)
    """
    # Load audio with error handling
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        raise ValueError(f"Error loading audio file: {str(e)}")
    
    # Compute STFT with optimized parameters
    n_fft = 2048  # Better frequency resolution
    hop_length = n_fft // 4  # Standard 25% overlap
    
    # Compute magnitude spectrogram and convert to dB
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Create figure with constrained layout
    plt.figure(figsize=(12, 6), dpi=100)
    
    # Display spectrogram with proper settings
    img = librosa.display.specshow(S_db, sr=sr, 
                                 hop_length=hop_length,
                                 x_axis='time', 
                                 y_axis='log',
                                 fmin=fmin,
                                 fmax=fmax,
                                 cmap='magma')  # Better color perception
    
    # Formatting
    plt.colorbar(img, format='%+2.0f dB', pad=0.02)
    plt.title(f'Spectrogram ({os.path.basename(audio_path)})', pad=10)
    plt.tight_layout()
    
    # Save with quality settings (removed quality parameter for PNG)
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def analyze_frequencies(audio_path, output_path):
    # Load audio with fixed sample rate and mono channel
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    
    # Compute spectrogram with proper parameters
    frequencies, times, spectrogram = signal.spectrogram(
        y, 
        fs=sr,
        nperseg=512,  # Frame size (adjust for resolution)
        noverlap=256  # Overlap (smooths transitions)
    )
    
    # Handle log scaling safely
    spectrogram_db = 10 * np.log10(spectrogram + 1e-12)  # Add small epsilon to avoid log(0)
    
    # Create the plot
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, frequencies, spectrogram_db, 
                  shading='auto',  # Better rendering
                  cmap='viridis')  # Better color map
    plt.ylim(0, 8000)  # Limit to human hearing range
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('Frequency Analysis') 
    # Save and clean up
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return frequencies as integers
    return frequencies.astype(int).tolist()

def detect_background_noise(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    noise_level = float(np.std(y))  # Convert to native Python float
    return noise_level

def detect_echo(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    corr = np.correlate(y, y, mode='full')
    echo_peak = int(np.argmax(corr))  # Convert to native Python int
    return echo_peak


# ==============Video=================
@app.route('/detect/video', methods=['POST'])
def detect_video_route():
    if 'file' not in request.files or 'user_id' not in request.form:
        return handle_error("No file or user ID provided", 400)
    
    file = request.files['file']
    user_id = request.form['user_id']

    if file.filename == '':
        return handle_error("No selected file", 400)

    # Save the file permanently
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        print(f"Serving Video: {filename} : {save_path} from {DETECTION_UPLOAD_FOLDER}")
        
        # Extract video metadata
        got_video_metadata = get_video_metadata(save_path)

        # Process the video using the ML model
        frames = preprocess_video(save_path)
        features = extract_features(frames, base_model)
        auxiliary_data = np.zeros((1, 20))
        predictions = video_model.predict([features, auxiliary_data])
        
        # Determine the label and confidence
        threshold = 0.5
        if predictions[0][1] > threshold:
            label = "FAKE"
            confidence = float(predictions[0][1])
        else:
            label = "REAL"
            confidence = float(predictions[0][0])

        # Analyze video frames with dynamic label and confidence
        frame_analysis_paths = analyze_video_frames(save_path, label, confidence)
        frame_analysis_json = json.dumps(frame_analysis_paths)  # Convert list to JSON string

        video_result = VideoDetectionResult(
            user_id=user_id,
            filename=filename,
            label=label,
            confidence=confidence,
            video_metadata=got_video_metadata,
            frame_analysis_image=frame_analysis_json,   
        )
        db.session.add(video_result)
        db.session.commit()
        
        # Save to the common Detection table
        detection_history  = DetectionHistory(
            user_id=user_id,
            detection_type='video',
            detection_id=video_result.id
        )

        db.session.add(detection_history)
        db.session.commit()

        # Return the response
        return jsonify({
            "id": video_result.id,
            "type": "video",
            "label": label,
            "confidence": confidence,
            "filename": filename,
            "metadata": got_video_metadata,
            "frame_analysis_url": frame_analysis_paths    # Relative path
        })
    except Exception as e:
        return handle_error(f"Error processing video: {str(e)}", 500)
    
def get_video_metadata(video_path):
    # Use exiftool to extract video metadata
    result = subprocess.run(
        ['D:/flask_server_fyp/ml_modals/exiftool.exe', '-json', video_path],  # Use -json flag
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        raise Exception(f"ExifTool error: {result.stderr.decode('utf-8')}")
    
    # Parse the JSON output
    video_metadata = json.loads(result.stdout.decode('utf-8'))
    return video_metadata[0]
import re
import shutil

def get_video_duration(video_path):
    metadata = {}
    duration = 0
 
    try:
        result = subprocess.run(
            ['D:/flask_server_fyp/ml_modals/exiftool.exe', '-json', video_path], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            metadata = json.loads(result.stdout)[0]
            duration_str = metadata.get('Duration', "0")

            # Extract duration from ExifTool output
            if ":" in duration_str:  # Format like "0:02:30.50"
                time_parts = [float(x) for x in duration_str.split(":")]
                duration = sum(x * 60 ** i for i, x in enumerate(reversed(time_parts)))
            else:
                duration = float(re.sub(r'[^\d.]', '', duration_str))

    except Exception as e:
        print(f"ExifTool failed: {e}") 
    if duration == 0:
        if shutil.which("ffmpeg") is None:
            raise Exception("FFmpeg is not installed or not found in system PATH")

        try:
            result = subprocess.run(
                ["ffmpeg", "-i", video_path, "-hide_banner"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr)
            if match:
                hours, minutes, seconds = map(float, match.groups())
                duration = hours * 3600 + minutes * 60 + seconds
        except Exception as e:
            print(f"FFmpeg failed: {e}")

    if duration == 0:
        raise Exception("Failed to extract video duration. The file may be corrupt or unsupported.")

    return metadata, duration


def analyze_video_frames(video_path, label, confidence):
    metadata, duration = get_video_duration(video_path)

    if duration == 0:
        raise Exception("Could not determine video duration.")

    num_frames = 5
    interval = duration / num_frames

    output_dir = os.path.join(app.config['DETECTION_UPLOAD_FOLDER'], "video_frames")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  

    saved_frames = []

    for i in range(1, num_frames + 1):
        capture_time = i * interval  
        frame_index = int(capture_time * fps)  

        if frame_index >= total_frames: 
            print(f"Skipping frame at {capture_time}s (out of range)")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            print(f"Frame at {capture_time}s could not be captured.")
            continue

         
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

         
        if label == "FAKE":
            box_color = (0, 0, 255)  # Red
            text = "Suspicious"
        elif label == "REAL" and confidence < 0.50:
            box_color = (0, 255, 255)  # Yellow
            text = "Low Confidence"
        else:
            box_color = (0, 255, 0)  # Green
            text = "Authentic"

         
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Ignore small areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

         
        timestamp = int(time.time())
        frame_filename = f'frame_{i}_{timestamp}.png'
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        saved_frames.append(frame_filename)

    cap.release()
    return saved_frames

# ======================
# ML PROCESSING FUNCTIONS
# ======================
def check_fake_or_real(file_path):
    img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = image_model.predict(img_array)
    prediction_value = float(prediction[0])  # Explicitly convert to float
    label = "Real" if prediction_value >= 0.5 else "Fake"
    return label, prediction_value

def preprocess_single_audio(file_path):
    audio_array, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
    features = np.mean(mfcc, axis=1)
    return np.expand_dims(features, axis=0)

def detect_audio(file_path):
    features = preprocess_single_audio(file_path)
    print("Feature shape:", features.shape)  # Debugging

    prediction = audio_model.predict(features)
    confidence = float(prediction[0][0])  # Ensure float extraction
    label = "Bonafide" if confidence > 0.5 else "Deepfake"

    return label, confidence

def preprocess_video(video_path, frame_count=20, target_size=(299, 299)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    while len(frames) < frame_count:
        frames.append(np.zeros((target_size[0], target_size[1], 3)))
    return np.array(frames) / 255.0

def extract_features(frames, base_model):
    batch_frames = frames.reshape(-1, 299, 299, 3)
    features = base_model.predict(batch_frames)
    return features.reshape(1, 20, 2048)

# SERPAPI_KEY = "f9ab863e8a2f061b60fbc78e42709507847324a15615a378b518936eef1702b9"  # Replace with your SerpAPI Key

# def search_similar_images(image_path):
#     try:
#         # Step 1: Construct the image URL
#         image_url = f"https://ddeb-182-186-51-130.ngrok-free.app/detection-files/{image_path}"
#         print(f"Image URL: {image_url}")  # Debugging

#         # Step 2: Test the image URL
#         import requests
#         response = requests.get(image_url)
#         if response.status_code != 200:
#             print(f"Image URL is not accessible. Status code: {response.status_code}")
#             return []

#         # Step 3: Prepare API request
#         search_params = {
#             "engine": "google_reverse_image",
#             "image_url": image_url,
#             "api_key": SERPAPI_KEY
#         }

#         # Step 4: Send request to SerpAPI
#         search = GoogleSearch(search_params)
#         search_results = search.get_dict()

#         # Debugging: Print the API response
#         print("API Response:", search_results)

#         # Step 5: Check if the response contains the expected data
#         if "images_results" not in search_results:
#             print("No 'images_results' key in API response.")
#             return []

#         # Step 6: Extract similar image URLs
#         similar_images = [img["original"] for img in search_results["images_results"][:5]]  # Get top 5 results
#         return similar_images

#     except Exception as e:
#         print(f"Error searching similar images: {str(e)}")
#         return []

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
    
    
    