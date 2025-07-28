import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# Clean corrupted images
def clean_dataset(data_dir='new_dataset'):
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print(f"Removing invalid image: {image_path}")
                    os.remove(image_path)
            except Exception as e:
                print(f"Issue with image {image_path}: {e}")

# Load and preprocess dataset
def load_dataset():
    data = tf.keras.utils.image_dataset_from_directory(
        'new_dataset', 
        image_size=(256, 256),
        batch_size=32
    )
    return data.map(lambda x, y: (x / 255, y))  # Normalize pixel values

# Build CNN model
def build_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
    return model

# Evaluate model and save predictions
def evaluate_model(model, test_data):
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    
    test_predictions = []
    test_images = []
    test_labels = []
    
    for batch in test_data.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X, verbose=0)
        
        # Store for analysis
        test_predictions.extend(yhat.flatten())
        test_images.extend(X)
        test_labels.extend(y)
        
        # Update metrics
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    
    print(f'\nEvaluation Metrics:')
    print(f'Precision: {pre.result().numpy():.4f}')
    print(f'Recall: {re.result().numpy():.4f}')
    print(f'Accuracy: {acc.result().numpy():.4f}')
    
    return test_predictions, test_images, test_labels

def main():
    # Clean dataset
    print("Cleaning dataset...")
    clean_dataset()
    
    # Load and split data
    print("\nLoading dataset...")
    data = load_dataset()
    
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    test_size = int(len(data) * 0.1) + 1
    
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)
    
    # Build and train model
    print("\nBuilding model...")
    model = build_model()
    
    print("\nTraining model...")
    model.fit(
        train,
        epochs=30,
        validation_data=val,
        verbose=1
    )
    
    # Evaluate and save predictions
    print("\nEvaluating model...")
    test_predictions, test_images, test_labels = evaluate_model(model, test)
    
    # Save model and predictions
    os.makedirs('models', exist_ok=True)
    model.save(os.path.join('models', 'soil_model.h5'))
    
    np.save(os.path.join('models', 'test_predictions.npy'), np.array(test_predictions))
    np.save(os.path.join('models', 'test_images.npy'), np.array(test_images))
    np.save(os.path.join('models', 'test_labels.npy'), np.array(test_labels))
    
    print("\nSaved model and test predictions to 'models' directory")

if __name__ == '__main__':
    main()