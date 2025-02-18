# Facial-Expression-Detector-
This script trains and runs a real-time facial emotion detection model. It preprocesses images, builds a CNN, and trains it using TensorFlow. If a model exists, it loads it for live detection via OpenCV, identifying faces and displaying emotions with emojis and confidence levels. Press 'Q' to exit live detection mode.
# emotion_detection_image_based.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# ======================
#  CONFIGURATION
# ======================
TRAIN_DIR = r"C:\Users\ASUS\Downloads\archive (1)\train"
TEST_DIR = r"C:\Users\ASUS\Downloads\archive (1)\test"
MODEL_PATH = r"C:\FER_Data\emotion_model_img.h5"
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 30

EMOTION_MAP = {
    0: ("Angry", "😠"),
    1: ("Disgust", "🤢"),
    2: ("Fear", "😨"),
    3: ("Happy", "😄"),
    4: ("Neutral", "😐"),
    5: ("Sad", "😢"),
    6: ("Surprise", "😲")
}

# ======================
#  DATA PREPARATION
# ======================
def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator

# ======================
#  MODEL ARCHITECTURE
# ======================
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.35),

        Conv2D(256, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.45),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

# ======================
#  REAL-TIME DETECTION
# ======================
class EmotionDetector:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def predict_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
            processed = face_roi.reshape(1, *IMG_SIZE, 1) / 255.0
            preds = self.model.predict(processed, verbose=0)[0]
            emotion_idx = np.argmax(preds)
            emotion, emoji = EMOTION_MAP[emotion_idx]
            confidence = preds[emotion_idx]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} {emoji} {confidence:.0%}",
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        return frame

# ======================
#  MAIN WORKFLOW
# ======================
def main():
    # Create data directory
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Data preparation
    train_gen, test_gen = create_generators()
    
    if not os.path.exists(MODEL_PATH):
        # Build and train model
        model = build_model((*IMG_SIZE, 1), train_gen.num_classes)
        model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=test_gen,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True),
                tf.keras.callbacks.EarlyStopping(patience=5)
            ]
        )
    else:
        print("✅ Using existing model")

    # Start detection
    detector = EmotionDetector(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    
    print("🎥 Live detection - Press Q to quit")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('Emotion Detection', detector.predict_emotion(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
