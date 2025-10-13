import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()
URL = os.getenv("URL")

def capture_video():
    cap = cv2.VideoCapture(0)


    cap = cv2.VideoCapture(f"{URL}")

    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()