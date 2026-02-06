import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- CONFIGURATION ---
DATASET_DIR = 'dataset'      # Folder containing your 4 activity subfolders
OUTPUT_DIR = 'output'        # Where the model will be saved
IMG_SIZE = (224, 224)        # MobileNetV3 requires 224x224 input
BATCH_SIZE = 32              # Number of images to process at once
EPOCHS = 30                  # Number of training rounds (increased to 30 for better accuracy)
LEARNING_RATE = 1e-4         # 0.0001 - Ideal for fine-tuning

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. DATA PREPARATION ---
# Data Augmentation: Creates variations of your images to prevent overfitting
# (e.g., flipping, slight rotations) which helps when you have a small dataset.
train_datagen = ImageDataGenerator(
    rescale=1.0/255,         # Normalize pixel values to 0-1
    rotation_range=20,       # Rotate slightly
    width_shift_range=0.2,   # Shift left/right
    height_shift_range=0.2,  # Shift up/down
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2     # Use 20% of images for validation (testing)
)

print("[INFO] Loading training data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("[INFO] Loading validation data...")
val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save the class names (e.g., 0=getting_up, 1=resting...) so we know them later
np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), train_generator.class_indices)
print(f"[INFO] Classes found: {train_generator.class_indices}")

# --- 2. BUILD MODEL (MobileNetV3-Small) ---
# We use MobileNetV3-Small as the backbone, as specified in your thesis for Raspberry Pi efficiency.
base_model = MobileNetV3Small(
    weights='imagenet',      # Pre-trained on ImageNet (Transfer Learning)
    include_top=False,       # Remove the top layer (we will add our own)
    input_shape=(224, 224, 3)
)

# Freeze the base model layers so we don't destroy the pre-trained patterns
base_model.trainable = False

# Add your custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)  # Dropout helps prevent overfitting
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 3. TRAIN MODEL ---
print("[INFO] Starting training...")
callbacks = [
    # Save the model only when accuracy improves
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "activity_model.h5"),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    # Stop early if the model stops learning to save time
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)

# --- 4. PLOT RESULTS ---
# Creates a graph showing how well the model learned
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plot_path = os.path.join(OUTPUT_DIR, "training_plot.png")
plt.savefig(plot_path)
print(f"[INFO] Training complete. Plot saved to {plot_path}")