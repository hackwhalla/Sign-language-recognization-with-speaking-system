import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import pickle
import pyttsx3
import time
import threading
import queue

# -------------------- CONFIG --------------------
offset = 20
imgSize = 300
labels = ["Doing", "Hai", "ok", "Whats"]
label_queue = queue.Queue()
prev_label = [None]
last_spoken_time = [0]  # shared mutable timestamp

# ------------------ SPEECH THREAD ------------------
def speech_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)

    while True:
        label = label_queue.get()
        if label == "__STOP__":
            break
        try:
            print("[Speaking]:", label)
            engine.say(label)
            engine.runAndWait()
        except Exception as e:
            print("Speech Error:", e)
        label_queue.task_done()

# ----------------- LOAD MODEL ------------------
with open('data.pkl', 'rb') as file:
    model = pickle.load(file)

# ----------------- DETECTION THREAD ------------------
def detect_and_predict():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.zeros((imgSize, imgSize, 3), np.uint8)

            try:
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap: hCal + hGap, :] = imgResize

                # Preprocess
                imgInput = imgWhite / 255.0
                imgInput = np.expand_dims(imgInput, axis=0)

                # Predict
                prediction = model.predict(imgInput)
                index = np.argmax(prediction)
                label = labels[index]

                print("Detected:", label)

                # Speak every 2 seconds even if same label
                current_time = time.time()
                if (label != prev_label[0]) or (current_time - last_spoken_time[0] > 2):
                    label_queue.put(label)
                    prev_label[0] = label
                    last_spoken_time[0] = current_time

                # Display
                cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                              (x - offset + 255, y - offset), (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, label, (x, y - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 4)

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

            except Exception as e:
                print("Error in detection:", e)

        cv2.imshow("Image", imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    label_queue.put("__STOP__")  # Stop speech thread

# ----------------- START BOTH THREADS ------------------

# Start speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Start detection (main thread)
detect_and_predict()

# Wait for speech thread to finish
speech_thread.join()


