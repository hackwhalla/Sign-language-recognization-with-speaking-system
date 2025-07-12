import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 50
imgSize = 300
counter = 0

folder = "Data/ok"

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to grab frame from camera")
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Get image size
        hImg, wImg, _ = img.shape

        # Boundary-safe cropping
        y1 = max(0, y - offset)
        y2 = min(hImg,y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(wImg, x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            print("⚠️ Skipping frame: empty crop due to boundary issue")
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        counter += 1
        save_path = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(save_path, imgWhite)
        print(f"✅ Saved: {save_path} | Count: {counter}")

    elif key == ord("q"):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
