import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
import os


# Image dimensions
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

# Paths to the dataset
TRAIN_DIR = 'pikachu_dataset/train/'
VALIDATION_DIR = 'pikachu_dataset/validation/'

if os.path.isfile("pikachu_classifier_model.h5") == False:
    # Data augmentation and preprocessing
    train_image_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    validation_image_gen = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')  # Binary classification

    validation_data_gen = validation_image_gen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary')

    # Building the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Train the model
    EPOCHS = 10
    history = model.fit(
        train_data_gen,
        steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_data_gen,
        validation_steps=validation_data_gen.samples // BATCH_SIZE
    )

    # Save the model
    model.save('pikachu_classifier_model.h5')
else:
    print("model already exists will now load model instead")
    model = load_model('pikachu_classifier_model.h5')
# Function to predict if an image is Pikachu or not
import numpy as np
from tensorflow.keras.preprocessing import image

def classify_pikachu(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return 'Pikachu' if prediction[0] > 0.5 else 'Not Pikachu'

"""
# get the path or directory
folder_dir = r"pikachu_dataset\test\not_pikachu"
folder_dir2 = "pikachu_dataset\test\pikachu"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        # Example usage:
        test_image_path = images
        result = classify_pikachu(test_image_path)
        print(f'This image is: {result}')

""" 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create a data generator for the test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Normalize the pixel values

# Set up the test data generator
test_generator = test_datagen.flow_from_directory(
    r'pikachu_dataset\test',  # Directory of your test data
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # Resize images
    batch_size=BATCH_SIZE,  # Set your batch size
    class_mode='binary',  # Use 'categorical' for multi-class
    shuffle=False  # Important for evaluation
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)

# Output the results
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Example usage:
test_image_path = r'pikachu_dataset\test\not_pikachu\not_pikachu_00531.jpg'
result = classify_pikachu(test_image_path)
print(f'This image is: {result}')

#example 2:
# Example usage:
test_image_path = r'pikachu_dataset\test\pikachu\pikachu_00530.jpg'
result = classify_pikachu(test_image_path)
print(f'This image is: {result}')