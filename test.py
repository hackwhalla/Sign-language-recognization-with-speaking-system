"""import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model.pkl", "Model/labels.txt")
offset = 20
imgSize = 300
counter = 0

labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                      cv2.FILLED)

        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)


import pickle
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
#from tensorflow.keras.models import load_model
import pyttsx3

# Initialize the engine
engine = pyttsx3.init()

# Set speaking rate (optional)
engine.setProperty('rate', 150)

# Set voice (optional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Load your trained Keras model
with open('data.pkl', 'rb') as file:
    model = pickle.load(file)

# Class labels (must be in same order as training)
labels = ["Doing", "Hai", "ok", "Whats"]

# Camera setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w

        try:
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

            # Preprocess input for model
            imgInput = imgWhite / 255.0
            imgInput = np.expand_dims(imgInput, axis=0)  # (1, 300, 300, 3)

            # Predict
            prediction = model.predict(imgInput)
            index = np.argmax(prediction)

            # Display
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                          (x - offset + 300, y - offset), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)

            engine.say(labels[index])
            #engine.runAndWait()

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print("Error in processing:", e)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)"""

#index = np.argmax(prediction)
#imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import pickle
import pyttsx3
import time
import threading

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Optional: Use a specific voice
    engine.say(label)
    engine.runAndWait()
    engine.stop()
    time.sleep(1)


# Initialize the engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Load the trained model from pickle
with open('data.pkl', 'rb') as file:
    model = pickle.load(file)

# Labels used during training
labels = ["Doing", "Hai", "ok", "Whats"]

# Camera setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
prev_label = ""

while True:
    success, img = cap.read()
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
            imgInput = np.expand_dims(imgInput, axis=0)  # (1, 300, 300, 3)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 0
            # Flatten the image if model expects 1D input (since it's from pickle)
            #imgInput = imgInput.reshape(1, -1)  # Convert to shape (1, 270000)

            # Predict
            prediction = model.predict(imgInput)
            index = np.argmax(prediction)
            print(index)
            print(prev_label)
            label = labels[index]
            if label != prev_label:
                threading.Thread(target=speak_text, args=(label,)).start()


            # Display
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                          (x - offset + 255, y - offset), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # Speak only when label changes to avoid repeating

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

        except Exception as e:
            print("Error in processing:", e)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
