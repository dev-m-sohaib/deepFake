# models.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(80), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)  # ✅ Rename to password_hash
    email = db.Column(db.String(120), nullable=False, unique=True)
    profile_picture = db.Column(db.String(255))

    def __init__(self, username, password, email, profile_picture):
        self.username = username
        self.password = password  # ✅ Use setter method to hash password
        self.email = email
        self.profile_picture = profile_picture

    @property
    def password(self):
        raise AttributeError("Password is not readable!")

    @password.setter
    def password(self, plain_text_password):
        self.password_hash = generate_password_hash(plain_text_password)  # ✅ Automatically hash password

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)  # ✅ Compare with hashed password

class DetectionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    detection_type = db.Column(db.String(50), nullable=False)  # 'image', 'audio', or 'video'
    detection_id = db.Column(db.Integer, nullable=False)  # ID from the specific detection table
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, detection_type, detection_id):
        self.user_id = user_id
        self.detection_type = detection_type
        self.detection_id = detection_id

class ImageDetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_metadata = db.Column(db.JSON)
    ela_image = db.Column(db.String(255))
    jpeg_image = db.Column(db.String(255))
    noise_image = db.Column(db.String(255))
    heatmap_image = db.Column(db.String(255))
    favorite = db.Column(db.Boolean, default=False)
    public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, filename, label, confidence, image_metadata=None, ela_image=None, jpeg_image=None, noise_image=None,heatmap_image=None, favorite=False, public=False):
        self.user_id = user_id
        self.filename = filename
        self.label = label
        self.confidence = confidence
        self.image_metadata = image_metadata
        self.ela_image = ela_image
        self.jpeg_image = jpeg_image
        self.heatmap_image = heatmap_image
        self.favorite = favorite
        self.public = public
        self.noise_image = noise_image

class AudioDetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    audio_metadata = db.Column(db.Text, nullable=True)  # Store metadata as JSON string
    spectrogram_path = db.Column(db.String(255), nullable=True)  # Path to spectrogram image
    frequency_graph_path = db.Column(db.String(255), nullable=True)  # Path to frequency graph image
    frequencies = db.Column(db.Text, nullable=True)  # Store frequency data as JSON string
    noise_level = db.Column(db.Float, nullable=True)  # Noise level detected in the audio
    echo_peak = db.Column(db.Float, nullable=True)  # Echo peak value detected
    favorite = db.Column(db.Boolean, default=False)
    public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Timestamp of detection

    def __init__(self, user_id, filename, label, confidence, audio_metadata=None, spectrogram_path=None, 
                 frequency_graph_path=None, frequencies=None, noise_level=None, echo_peak=None, favorite=False, public=False):
        self.user_id = user_id
        self.filename = filename
        self.label = label
        self.confidence = confidence
        self.audio_metadata = audio_metadata
        self.spectrogram_path = spectrogram_path
        self.frequency_graph_path = frequency_graph_path
        self.frequencies = frequencies
        self.noise_level = noise_level
        self.favorite = favorite
        self.public = public
        self.echo_peak = echo_peak

class VideoDetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    label = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    video_metadata = db.Column(db.JSON)
    frame_analysis_image = db.Column(db.String(255))
    favorite = db.Column(db.Boolean, default=False)
    public = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, filename, label, confidence, video_metadata=None, frame_analysis_image=None, favorite=False, public=False):
        self.user_id = user_id
        self.filename = filename
        self.label = label
        self.confidence = confidence
        self.video_metadata = video_metadata
        self.favorite = favorite
        self.public = public
        self.frame_analysis_image = frame_analysis_image
        
class UserFavourties(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    detection_type = db.Column(db.String(255), nullable=False)
    detection_id = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, detection_type, detection_id):
        self.user_id = user_id
        self.detection_type = detection_type
        self.detection_id = detection_id

class PublicDetections(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    detection_type = db.Column(db.String(255), nullable=False)
    detection_id = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, detection_type, detection_id):
        self.user_id = user_id
        self.detection_type = detection_type
        self.detection_id = detection_id



# class DetectionResult(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_id = db.Column(db.Integer, nullable=False)
#     filename = db.Column(db.String(255), nullable=False)
#     label = db.Column(db.String(50), nullable=False)
#     confidence = db.Column(db.Float, nullable=False)  # Ensure this is Float
#     detection_type = db.Column(db.String(50), nullable=False)
#     image_metadata = db.Column(db.JSON)
#     video_metadata = db.Column(db.JSON)
#     ela_image = db.Column(db.String(255))
#     jpeg_image = db.Column(db.String(255))
#     noise_image = db.Column(db.String(255))
#     frame_analysis_image = db.Column(db.String(255))
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

#     def __init__(self, user_id, filename, label, confidence, detection_type, image_metadata, ela_image, jpeg_image, noise_image, frame_analysis_image, video_metadata):
#         self.user_id = user_id
#         self.filename = filename
#         self.label = label
#         self.confidence = confidence
#         self.detection_type = detection_type
#         self.image_metadata = image_metadata
#         self.ela_image = ela_image
#         self.noise_image = noise_image
#         self.jpeg_image = jpeg_image
#         self.frame_analysis_image = frame_analysis_image
#         self.video_metadata = video_metadata
        