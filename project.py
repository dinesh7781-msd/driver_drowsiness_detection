import cv2
import dlib
import numpy as np
from flask import Flask, send_file, jsonify
import threading
import serial
import time
from scipy.spatial import distance
import winsound
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

app = Flask(__name__)

camera_running = False
status = "Idle"
drowsy_count = 0
yawn_count = 0
# Load pre-trained CNN model for face recognition
# Build CNN for drowsiness classification
def build_drowsiness_model():
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = False
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

drowsiness_model = build_drowsiness_model()
logs = []

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def yawn_ratio(mouth):
    vertical = distance.euclidean(mouth[2], mouth[6])  
    horizontal = distance.euclidean(mouth[0], mouth[4])  
    return vertical / horizontal

def head_tilt_angle(landmarks):
    left = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                     (landmarks.part(39).x, landmarks.part(39).y)])
    right = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                      (landmarks.part(45).x, landmarks.part(45).y)])

    left_center = left.mean(axis=0)
    right_center = right.mean(axis=0)

    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]

    return np.degrees(np.arctan2(dY, dX))

def init_serial(port='COM3', baudrate=9600):
    try:
        return serial.Serial(port, baudrate, timeout=1)
    except:
        print("Serial connection failed")
        return None

def vibrate(ser):
    if ser:
        ser.write(b'1')
        time.sleep(0.3)
        ser.write(b'0')

def run_detection():
    global camera_running, status, drowsy_count, yawn_count, logs

    cap = cv2.VideoCapture(0)
    ser = init_serial()

    try:
        eye_thresh = 0.25
        tilt_thresh = 15

        YAWN_RATIO_THRESH = 0.6

        eye_closed_start = None

        status = "Monitoring"

        while camera_running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)

                left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                if ear < eye_thresh:
                    if eye_closed_start is None:
                        eye_closed_start = time.time()
                    elif time.time() - eye_closed_start >= 2:
                        drowsy_count += 1
                        logs.append("😴 Eyes closed > 2 sec")

                        cv2.putText(frame, "DROWSY!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, "STOP THE VEHICLE ASIDE", (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(frame, "YOU ARE LOOKING TIRED", (10, 110),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.putText(frame, "TAKE SOME REST", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                        winsound.Beep(1000, 300)
                        vibrate(ser)

                        eye_closed_start = None
                else:
                    eye_closed_start = None

                angle = head_tilt_angle(landmarks)

                if abs(angle) > tilt_thresh:
                    logs.append(f"📐 Head Tilt Alert: {angle:.1f}")
                    cv2.putText(frame, f"TILT ALERT {angle:.1f}", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    winsound.Beep(700, 300)
                    vibrate(ser)

                mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(60, 68)])
                cv2.polylines(frame, [mouth], True, (0, 255, 255), 2)

                ratio = yawn_ratio(mouth)

                if ratio > YAWN_RATIO_THRESH:
                    yawn_count += 1
                    logs.append("🥱 Yawn detected (instant)")

                    cv2.putText(frame, "YAWN!", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

                    winsound.Beep(800, 300)

            cv2.imshow("Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if ser:
            ser.close()
        status = "Stopped"

@app.route('/')
def home():
    return send_file('templates/frontend.html')

@app.route('/start')
def start():
    global camera_running
    if not camera_running:
        camera_running = True
        threading.Thread(target=run_detection).start()
    return "Started"

@app.route('/stop')
def stop():
    global camera_running
    camera_running = False
    return "Stopped"

@app.route('/status')
def get_status():
    return jsonify({
        "status": status,
        "count": drowsy_count,
        "yawn": yawn_count,
        "logs": logs[-5:]
    })

if __name__ == '__main__':
    app.run(debug=True)