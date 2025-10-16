import cv2
import sys

def process_video(source: str | int, callback=None) -> None:
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        sys.stderr.write(f"Error: Could not open video source: {source}")
        return

    try:
        print("Successfully opened video stream. Press 'q' to quit.")
        while True:
            ret, frame = cap.read()

            if not ret:
                print("End of video stream.")
                break

            display_frame = callback(frame) if callback else frame

            cv2.imshow("Video Stream", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, exiting.")
                break
    finally:
        print("Releasing video resources...")
        cap.release()
        cv2.destroyAllWindows()