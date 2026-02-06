# scripts/01_train_improved.py

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- CONFIGURATION ---
DATASET_DIR = 'dataset'
OUTPUT_DIR = 'output'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16              # Reduced batch size to help generalization
EPOCHS = 50                  # Increased epochs
LEARNING_RATE = 1e-3         # Higher initial learning rate (0.001)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. DATA PREPARATION ---
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,       # Reduced rotation slightly to keep postures clear
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
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

np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), train_generator.class_indices)

# --- 2. BUILD MODEL (Fine-Tuning) ---
base_model = MobileNetV3Small(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# UNFREEZE the last 30 layers so they can "learn" your specific dataset
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)  # Helps stabilize training
x = Dense(256, activation='relu')(x) # Increased neuron count
x = Dropout(0.4)(x)          # Increased dropout
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. COMPILE & TRAIN ---
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("[INFO] Starting FINE-TUNING training...")
callbacks = [
    # CRITICAL FIX: Changed .h5 to .keras
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "activity_model.keras"), 
        monitor="val_accuracy", 
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,             # More patience before stopping
        restore_best_weights=True,
        verbose=1
    ),
    # Slow down learning rate if we get stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
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

# --- 4. PLOT ---
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

plt.savefig(os.path.join(OUTPUT_DIR, "training_plot.png"))
print("[INFO] Done.")