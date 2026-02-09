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
from tensorflow.keras.regularizers import l2  # <--- NEW IMPORT

DATASET_DIR = 'dataset'
OUTPUT_DIR = 'output'
IMG_SIZE = (224, 224)
BATCH_SIZE = 16              
EPOCHS = 50                 
LEARNING_RATE = 1e-3   # Higher LR is okay when the base is frozen

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. DATA GENERATORS ---
train_datagen = ImageDataGenerator(
    rotation_range=20,       # Increased rotation slightly
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
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
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), train_generator.class_indices)

# --- 2. BUILD MODEL (Transfer Learning - Anti-Overfitting Mode) ---
base_model = MobileNetV3Small(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze the entire base model
base_model.trainable = False 

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)

# CHANGE 1: Smaller "Brain" (64 neurons instead of 256)
# CHANGE 2: Added L2 Regularization to punish memorization
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)

# CHANGE 3: Increased Dropout to 0.6 (High difficulty)
x = Dropout(0.6)(x) 

predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 3. COMPILE & TRAIN ---
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(f"[INFO] Training (Base Frozen + Anti-Overfitting)...")

callbacks = [
    ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "activity_model.keras"), 
        monitor="val_accuracy", 
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=10,        # Give it time to settle
        restore_best_weights=True,
        verbose=1
    ),
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