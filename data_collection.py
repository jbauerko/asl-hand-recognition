import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

dataDir = os.path.join("data","B")
count = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]

        # Create an image with just the hand
        x, y, w, h =  hand['bbox']
        imgCropped = img[y - offset:y + h + offset, x - offset:x + w + offset]
        
        # Create an image that will adapt based on the size of the bounding box of the hand
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCropShape = imgCropped.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            newWidth = math.ceil(k * w) 
            imgResize = cv2.resize(imgCropped, (newWidth, imgSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imgSize - newWidth)/2)
            imgWhite[:, widthGap:newWidth+widthGap] = imgResize 
        else: 
            k = imgSize / w
            newHeight = math.ceil(k * h) 
            imgResize = cv2.resize(imgCropped, (imgSize, newHeight))
            imgResizeShape = imgResize.shape
            heightGap = math.ceil((imgSize - newHeight) / 2)
            imgWhite[heightGap:newHeight+heightGap, :] = imgResize 

        cv2.imshow("ImageCrop", imgCropped)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        count += 1
        cv2.imwrite(f'{dataDir}/Image_{time.time()}.jpg', imgWhite)
        print(count)