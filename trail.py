import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet169
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)

# Enhanced model parameters
IMG_HEIGHT = 224  # Increased for better feature extraction
IMG_WIDTH = 224
BATCH_SIZE = 8    # Smaller batch size for better generalization
EPOCHS = 150
NUM_CLASSES = 3

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.2
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Create data generators with data balancing
train_generator = train_datagen.flow_from_directory(
    'C:/Stroke Detection/dataset/Stroke_classification',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    'C:/Stroke Detection/dataset/Stroke_classification',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

def create_advanced_model():
    # Base model: DenseNet169 (better feature extraction)
    base_model = DenseNet169(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Unfreeze layers for fine-tuning
    for layer in base_model.layers[-60:]:  # Fine-tune more layers
        layer.trainable = True

    # Create custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # First dense block
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Second dense block
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Third dense block
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer with softmax
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Learning rate scheduler with warmup
def cosine_decay_with_warmup(epoch):
    warmup_epochs = 5
    total_epochs = EPOCHS
    warmup_lr = 0.0001
    base_lr = 0.001
    
    if epoch < warmup_epochs:
        # Linear warmup
        return float(warmup_lr + (base_lr - warmup_lr) * (epoch / warmup_epochs))
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return float(base_lr * (0.5 * (1 + np.cos(np.pi * progress))))

# Create and compile model
model = create_advanced_model()

# Compile with advanced settings
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.0001
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Enhanced callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        min_delta=0.001
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.LearningRateScheduler(cosine_decay_with_warmup, verbose=1)
]

# Calculate class weights
total_samples = sum([len(train_generator.filenames)])
class_counts = [len([f for f in train_generator.filenames if f.startswith(c)]) 
                for c in train_generator.class_indices.keys()]
class_weights = {i: total_samples / (len(class_counts) * count) 
                for i, count in enumerate(class_counts)}

# Train with mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights
)

# Save the final model
model.save('detection_model.keras')

# Enhanced prediction function with ensemble predictions
def predict_image(model_path, image_path, confidence_threshold=0.7):
    loaded_model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess image
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    
    # Create multiple augmented versions
    predictions = []
    # Original image
    predictions.append(loaded_model.predict(np.expand_dims(img_array, axis=0)))
    
    # Augmented versions
    for _ in range(4):
        augmented = train_datagen.random_transform(img_array)
        predictions.append(loaded_model.predict(np.expand_dims(augmented, axis=0)))
    
    # Horizontal flip
    flipped = np.fliplr(img_array)
    predictions.append(loaded_model.predict(np.expand_dims(flipped, axis=0)))
    
    # Average predictions
    final_pred = np.mean(predictions, axis=0)
    max_confidence = float(np.max(final_pred))
    predicted_class = ['Haemorrhagic', 'Ischemic', 'Normal'][np.argmax(final_pred)]
    
    if max_confidence < confidence_threshold:
        return "Uncertain", max_confidence * 100
    
    return predicted_class, max_confidence * 100

# Plot training history
def plot_training_history(history):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

plot_training_history(history)
"""
print("\nTraining completed!")
print("Models saved as:")
print("- 'stroke_detection_model.keras' (final model)")
print("- 'best_model.keras' (best performing model during training)")
print("- 'training_history.png' (training plots)")
"""