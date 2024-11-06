import os
import gc
import math
import sys
import cv2
import pydicom
import numpy as np
import pandas as pd
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras import layers, models

print("Debug Information:")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA_HOME: {os.getenv('CUDA_HOME')}")
print(f"LD_LIBRARY_PATH: {os.getenv('LD_LIBRARY_PATH')}")
print("CUDA devices:", tf.config.list_physical_devices('GPU'))
print("CUDA built with:", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])

def setup_gpu():
    # Clear any existing GPU memory
    tf.keras.backend.clear_session()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU and limit memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set a more conservative memory limit (adjust based on your GPU)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 10)]  # 10GB
            )
            
            # Enable mixed precision globally
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"Found {len(gpus)} GPU(s)")
            print(f"Memory growth enabled: {tf.config.experimental.get_memory_growth(gpus[0])}")
            print("GPU configuration completed successfully")
            
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPUs found")
    return gpus

def verify_data_paths(base_dir, train_dir, test_dir):
    print(f"Base dir: {os.path.abspath(base_dir)}")  
    print(f"Train dir: {os.path.abspath(train_dir)}")
    print(f"Test dir: {os.path.abspath(test_dir)}")
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise ValueError(f"Test directory not found: {test_dir}")

def analyze_class_distribution(df):
    if 'target' not in df.columns:
        raise ValueError("DataFrame must contain 'target' column")
    
    distribution = df['target'].value_counts(normalize=True)
    print("\nClass Distribution:")
    print(distribution)
    print("\nTotal samples:", len(df))
    print("Positive samples:", len(df[df['target'] == 1]))
    print("Negative samples:", len(df[df['target'] == 0]))
    return distribution

def compute_class_weights(y):
    """Compute class weights for imbalanced dataset"""
    total = len(y)
    pos = np.sum(y == 1)
    neg = total - pos
    return {0: total/(2*neg), 1: total/(2*pos)}

@tf.function
def process_image(img):
    img = tf.cast(img, tf.float32)
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img) + 1e-7)
    img = img * 255.0
    return img

def dicom_to_array(file_path):
    try:
        dicom = pydicom.dcmread(file_path)
        img = dicom.pixel_array
        img = cv2.resize(img, (224, 224))
        
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.squeeze(img, axis=2)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] > 3:
            img = img[..., :3]
            
        img = process_image(img)
        return img.numpy() if isinstance(img, tf.Tensor) else img
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def prepare_metadata(dataframe, is_train=True):
    dataframe = dataframe.copy()
    anatom_site_col = "anatom_site_general_challenge"
    
    dataframe['sex'] = dataframe['sex'].fillna('unknown')
    dataframe[anatom_site_col] = dataframe[anatom_site_col].fillna('unknown')
    dataframe['age_approx'] = dataframe['age_approx'].fillna(dataframe['age_approx'].median())
    
    categorical_cols = ['sex', anatom_site_col]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(dataframe[categorical_cols])
    
    numerical_cols = ['age_approx']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(dataframe[numerical_cols].values.reshape(-1, 1))
    
    return np.hstack([encoded_features, scaled_features])

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, images_dir, metadata, batch_size=16, shuffle=True, augment=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_filenames = self.dataframe['filename'].values
        self.labels = self.dataframe['target'].values.astype(np.float32)
        self.images_dir = os.path.abspath(images_dir)
        self.metadata = metadata
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        if not os.path.exists(self.images_dir):
            raise ValueError(f"Image directory not found: {self.images_dir}")
        
        self.valid_indexes = self._validate_files()
        if len(self.valid_indexes) == 0:
            raise ValueError("No valid images found in directory")
            
        self.on_epoch_end()
        
        if self.augment:
            self.augmentor = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomContrast(0.2),
            ])

    def _validate_files(self):
        valid_indexes = []
        for idx, fname in enumerate(self.image_filenames):
            if os.path.exists(os.path.join(self.images_dir, fname)):
                valid_indexes.append(idx)
        return np.array(valid_indexes)

    def __len__(self):
        return math.ceil(len(self.valid_indexes) / self.batch_size)

    def __getitem__(self, idx):
        try:
            with tf.device('/CPU:0'):
                start_idx = idx * self.batch_size
                end_idx = min((idx + 1) * self.batch_size, len(self.valid_indexes))
                batch_indexes = self.valid_indexes[start_idx:end_idx]
                
                batch_images = []
                batch_labels = []
                batch_metadata = []
                
                for index in batch_indexes:
                    img_path = os.path.join(self.images_dir, self.image_filenames[index])
                    img = dicom_to_array(img_path)
                    
                    if img is not None:
                        if self.augment:
                            img = self.augmentor(tf.convert_to_tensor(img[np.newaxis, ...]), training=True)[0]
                        batch_images.append(img / 255.0)
                        batch_labels.append(self.labels[index])
                        batch_metadata.append(self.metadata[index])
                
                if not batch_images:
                    return {
                        'image_input': np.zeros((1, 224, 224, 3), dtype=np.float32),
                        'metadata_input': np.zeros((1, self.metadata.shape[1]), dtype=np.float32)
                    }, np.array([0.0], dtype=np.float32)
                
                batch_images = np.array(batch_images, dtype=np.float32)
                batch_metadata = np.array(batch_metadata, dtype=np.float32)
                batch_labels = np.array(batch_labels, dtype=np.float32)
                
                return {
                    'image_input': batch_images,
                    'metadata_input': batch_metadata
                }, batch_labels

        except Exception as e:
            print(f"Error in batch {idx}: {str(e)}")
            return {
                'image_input': np.zeros((1, 224, 224, 3), dtype=np.float32),
                'metadata_input': np.zeros((1, self.metadata.shape[1]), dtype=np.float32)
            }, np.array([0.0], dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.valid_indexes)

def create_model(metadata_shape):
    # Image input branch
    image_input = layers.Input(shape=(224, 224, 3), name='image_input')
    
    # Initial convolution block with L2 regularization
    x = layers.Conv2D(64, 7, strides=2, padding='same', 
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    def residual_block(x, filters, stride=1):
        shortcut = x
        
        x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 3, padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x
    
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 256, stride=2)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Metadata branch with L2 regularization
    metadata_input = layers.Input(shape=(metadata_shape,), name='metadata_input')
    y = layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(metadata_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    
    # Combine branches
    combined = layers.concatenate([x, y])
    z = layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(1, activation='sigmoid')(z)
    
    model = models.Model(inputs=[image_input, metadata_input], outputs=output)
    
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
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
    
    return model

class ModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, base_output_dir):
        super().__init__()
        self.base_output_dir = base_output_dir
        
    def on_epoch_end(self, epoch, logs=None):
        try:
            # Create a directory for this epoch
            epoch_dir = os.path.join(self.base_output_dir, f'epoch_{epoch + 1}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Save the model
            model_path = os.path.join(epoch_dir, 'model.h5')
            self.model.save(model_path)
            
            # Save the metrics for this epoch
            if logs is not None:
                metrics_path = os.path.join(epoch_dir, 'metrics.txt')
                with open(metrics_path, 'w') as f:
                    for metric, value in logs.items():
                        f.write(f"{metric}: {value}\n")
            
            print(f"\nSaved model and metrics for epoch {epoch + 1}")
        except Exception as e:
            print(f"Error saving model for epoch {epoch + 1}: {str(e)}")

def plot_metrics(history, output_dir):
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 3, i)
        plt.plot(history.history[metric], label=f'Training {metric}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300)
    plt.close()

def check_system_resources():
    """Check available system resources before training"""
    try:
        import psutil
        print("\nSystem Resources:")
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
        print(f"Free Disk Space: {psutil.disk_usage('/').free / (1024**3):.2f} GB")
    except ImportError:
        print("psutil not installed - skipping system resource check")

def train_model(train_generator, val_generator, class_weights, epochs=30):
    metadata_shape = train_generator.metadata.shape[1]
    model = create_model(metadata_shape)
    
    # Create timestamp-based directory for all outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"training_output_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    callbacks = [
        ModelSaver(base_output_dir),  # Updated to save each epoch
        tf.keras.callbacks.CSVLogger(os.path.join(base_output_dir, 'training_log.csv')),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        max_queue_size=10,
        workers=4,
        use_multiprocessing=False
    )
    
    # Save final model
    final_model_path = os.path.join(base_output_dir, 'final_model.h5')
    model.save(final_model_path)
    
    # Plot and save metrics
    plot_metrics(history, base_output_dir)
    
    return model, history

if __name__ == "__main__":
    try:
        check_system_resources()
        
        gpus = setup_gpu()
        BASE_DIR = os.path.expanduser(".")
        TRAIN_DIR = os.path.join(BASE_DIR, 'train')
        TEST_DIR = os.path.join(BASE_DIR, 'test')
        print('=============================================================================')
        print(f"test dir {TEST_DIR}\ntrain dir {TRAIN_DIR}")
        print('=============================================================================')
        verify_data_paths(BASE_DIR, TRAIN_DIR, TEST_DIR)
        
        print("\nStarting data loading and preprocessing...")
        
        train_csv = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))
        test_csv = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))
        
        train_csv['filename'] = train_csv['image_name'].apply(lambda x: f"{x}.dcm")
        test_csv['filename'] = test_csv['image'].apply(lambda x: f"{x}.dcm")
        train_csv['target'] = train_csv['target'].astype(np.float32)
        
        print("\nData loading completed")
        
        analyze_class_distribution(train_csv)
        
        train_df, val_df = train_test_split(
            train_csv, 
            test_size=0.2, 
            random_state=42, 
            stratify=train_csv['target']
        )
        
        print("\nPreparing metadata...")
        
        train_metadata = prepare_metadata(train_df, is_train=True)
        val_metadata = prepare_metadata(val_df, is_train=False)
        
        class_weights = compute_class_weights(train_df['target'])
        print("\nClass weights:", class_weights)
        
        print("\nInitializing data generators...")
        
        train_generator = DataGenerator(
            train_df, 
            TRAIN_DIR,
            train_metadata, 
            batch_size=32, 
            shuffle=True,
            augment=True
        )

        val_generator = DataGenerator(
            val_df, 
            TRAIN_DIR,
            val_metadata, 
            batch_size=32, 
            shuffle=False,
            augment=False
        )
        
        print("\nStarting model training...")
        print(f"Training on {len(train_df)} samples")
        print(f"Validating on {len(val_df)} samples")
        
        model, history = train_model(train_generator, val_generator, class_weights, epochs=30)
        
        history_df = pd.DataFrame(history.history)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_df.to_csv(f'training_history_{timestamp}.csv', index=False)
        
        print("\nTraining completed successfully!")
        print(f"Results saved with timestamp: {timestamp}")
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            gc.collect()
            tf.keras.backend.clear_session()
        except:
            pass
