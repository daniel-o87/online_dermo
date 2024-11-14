from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import tensorflow as tf
import pydicom
import cv2
import io
import os
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import uvicorn
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def setup_tensorflow():
    """Configure TensorFlow settings"""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.info("No GPU found. Running on CPU.")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            logger.info(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        logger.warning(f"Error configuring TensorFlow: {e}")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = FastAPI(title="Medical Image Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and preprocessing
model = None
encoder = None
scaler = None

class MetadataInput(BaseModel):
    sex: str
    anatom_site_general_challenge: str
    age_approx: float

class PredictionResponse(BaseModel):
    prediction: float
    prediction_probability: float
    processing_time_ms: float

def create_model(metadata_shape):
    """Recreate the model architecture"""
    # Image input branch
    image_input = tf.keras.layers.Input(shape=(224, 224, 3), name='image_input')
    
    # Initial convolution block
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', 
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(image_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    def residual_block(x, filters, stride=1):
        shortcut = x
        
        x = tf.keras.layers.Conv2D(filters, 3, strides=stride, padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(filters, 3, padding='same',
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(shortcut)
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)
        
        x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Metadata branch
    metadata_input = tf.keras.layers.Input(shape=(metadata_shape,), name='metadata_input')
    y = tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(metadata_input)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dropout(0.3)(y)
    
    # Combine branches
    combined = tf.keras.layers.concatenate([x, y])
    z = tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Dropout(0.3)(z)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(z)
    
    return tf.keras.Model(inputs=[image_input, metadata_input], outputs=output)

def initialize_preprocessors():
    """Initialize preprocessors with the same settings as training"""
    global encoder, scaler
    
    # Initialize encoders with known categories
    sex_categories = ['male', 'female', 'unknown']
    anatomic_site_categories = [
        'head/neck', 'upper extremity', 'lower extremity', 
        'torso', 'palms/soles', 'oral/genital', 'unknown'
    ]
    
    # Create sample data for fitting
    sample_data = []
    for sex in sex_categories:
        for site in anatomic_site_categories:
            sample_data.append([sex, site])
    
    # Initialize and fit encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(sample_data)
    
    # Calculate metadata shape
    metadata_shape = len(sex_categories) + len(anatomic_site_categories) + 1  # +1 for age
    
    # Initialize and fit scaler with reasonable age range
    scaler = StandardScaler()
    age_range = np.linspace(0, 100, 1000).reshape(-1, 1)
    scaler.fit(age_range)
    
    return metadata_shape

def load_model():
    """Load the trained model and initialize preprocessors"""
    global model
    try:
        # Initialize preprocessors first
        metadata_shape = initialize_preprocessors()
        logger.info("Preprocessors initialized successfully")
        
        # Create fresh model
        model = create_model(metadata_shape)
        logger.info("Model architecture created successfully")
        
        # Get model weights path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model_iterations/attempt_1', 'final_model.h5')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        logger.info(f"Loading weights from {model_path}")
        model.load_weights(model_path)
        logger.info("Model weights loaded successfully")
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@tf.function
def process_image(img):
    """Process image using same function as training"""
    img = tf.cast(img, tf.float32)
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img) + 1e-7)
    img = img * 255.0
    return img

def prepare_metadata(metadata_dict: dict):
    """Prepare metadata using same preprocessing as training"""
    try:
        # Handle missing values same as training
        sex = metadata_dict.get('sex', 'unknown')
        site = metadata_dict.get('anatom_site_general_challenge', 'unknown')
        age = float(metadata_dict.get('age_approx', 45.0))  # Use median age as default
        
        categorical_data = np.array([[sex, site]])
        encoded_categorical = encoder.transform(categorical_data)
        
        numerical_data = np.array([[age]])
        scaled_numerical = scaler.transform(numerical_data)
        
        return np.hstack([encoded_categorical, scaled_numerical])
        
    except Exception as e:
        logger.error(f"Error processing metadata: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Error processing metadata: {str(e)}")

def process_dicom(file_contents: bytes):
    """Process DICOM file using same preprocessing as training"""
    try:
        # Read DICOM from bytes
        dataset = pydicom.dcmread(io.BytesIO(file_contents))
        
        # Check if decompression is needed
        if dataset.file_meta.TransferSyntaxUID.is_compressed:
            logger.info("Compressed DICOM detected, decompressing...")
            try:
                # Try using GDCM first
                dataset.decompress()
            except Exception as e:
                logger.warning(f"GDCM decompression failed: {e}, trying PIL...")
                try:
                    # Try using Pillow as fallback
                    dataset.decompress('pillow')
                except Exception as e:
                    logger.error(f"All decompression attempts failed: {e}")
                    raise
        
        # Get pixel array
        img = dataset.pixel_array
        
        # Handle different photometric interpretations
        if hasattr(dataset, 'PhotometricInterpretation'):
            pi = dataset.PhotometricInterpretation
            if pi == "MONOCHROME1":
                img = np.invert(img)
            elif pi == "MONOCHROME2":
                pass  # Default interpretation
            logger.info(f"Processed PhotometricInterpretation: {pi}")
        
        # Resize to target size
        img = cv2.resize(img, (224, 224))
        
        # Convert to RGB if necessary
        if img.ndim == 2:
            logger.info("Converting grayscale to RGB")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            logger.info("Converting single channel to RGB")
            img = np.squeeze(img, axis=2)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] > 3:
            logger.info("Converting multi-channel to RGB")
            img = img[..., :3]
        
        # Process image using training pipeline
        img = process_image(img)
        img = img.numpy() if isinstance(img, tf.Tensor) else img
        img = img / 255.0
        
        logger.info("DICOM processing completed successfully")
        return img
        
    except Exception as e:
        logger.error(f"Error processing DICOM image: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Error processing DICOM image: {str(e)}\n"
                   "Please ensure the DICOM file is valid and not corrupted."
        )

@app.on_event("startup")
async def startup_event():
    """Initialize model and preprocessors on startup"""
    setup_tensorflow()
    if not load_model():
        logger.error("Failed to load model and preprocessors")
        raise RuntimeError("Failed to load model and preprocessors")
    logger.info("Application startup completed successfully")

@app.get("/ping")
def pong():
    """Health check endpoint"""
    return {"ping": "pong!"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    metadata: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Predict from DICOM image and metadata
    """
    import time
    start_time = time.time()
    
    try:
        metadata_dict = json.loads(metadata)
        
        file_contents = await file.read()
        image = process_dicom(file_contents)
        image = np.expand_dims(image, axis=0)
        
        metadata_processed = prepare_metadata(metadata_dict)
        
        prediction = model.predict(
            {
                'image_input': image,
                'metadata_input': metadata_processed
            },
            verbose=0
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=float(prediction[0][0] > 0.5),
            prediction_probability=float(prediction[0][0]),
            processing_time_ms=processing_time
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Invalid metadata JSON format")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8008, reload=True)
