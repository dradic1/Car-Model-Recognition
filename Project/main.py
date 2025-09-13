from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications import VGG16
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, Adamax

IMAGE_SIZE = 256
channels = 3
img_shape = (IMAGE_SIZE, IMAGE_SIZE, channels)

train_datadir = './Projektni zadatak/Cars Dataset/train'
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    horizontal_flip=True,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    train_datadir,
    target_size=(256, 256),
    class_mode="categorical",
    shuffle=True,
    batch_size=16,
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_datadir,
    target_size=(256, 256),
    class_mode="categorical",
    shuffle=False,
    batch_size=16,
    subset='validation'
)

test_datadir = './Projektni zadatak/Cars Dataset/test'
test_datagen = ImageDataGenerator(
    # rescale=1./255
)
test_generator = test_datagen.flow_from_directory(
    test_datadir,
    target_size=(256, 256),
    class_mode="categorical",
    shuffle=False,
    batch_size=16
)

class_names = list(train_generator.class_indices.keys())
print(class_names)


base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', input_shape=img_shape, pooling='max')
# base_model.trainable = False
model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(len(class_names), activation='softmax')
])

model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

earlystop = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.01,
    patience=5,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

log_dir = "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    train_generator,
    epochs=25,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=None,
    shuffle=True,
    callbacks=[checkpoint, earlystop, reduce_lr, tensorboard_callback]
)

evaluation = model.evaluate(test_generator)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

validation_generator.reset()
predictions = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

# Izrada matrice konfuzije
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report
y_true=test_generator.classes
predictions=model.predict(test_generator,  steps =813)
y_pred = np.argmax(predictions, axis=-1)
print(classification_report(y_true,y_pred))
