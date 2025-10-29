import cv2
import os

ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
MARKER_SIZE_PIXELS = 400
NUM_MARKERS = 4
OUTPUT_DIR = "../assets/aruco_markers"


def create_markers():
    """
    Generates and saves ArUco markers as image files.
    """
    print(f"Generating {NUM_MARKERS} ArUco markers from dictionary '{ARUCO_DICT_NAME}'...")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    for marker_id in range(NUM_MARKERS):
        marker_image = cv2.aruco.generateImageMarker(
            ARUCO_DICT,
            marker_id,
            MARKER_SIZE_PIXELS
        )

        file_name = os.path.join(OUTPUT_DIR, f"aruco_marker_{marker_id}.png")
        cv2.imwrite(file_name, marker_image)
        print(f"  - Saved marker with ID {marker_id} to {file_name}")

    print(f"\nDone! You can find the markers in the '{OUTPUT_DIR}/' directory.")
    print("Print them and place them on the corners of your game board.")


if __name__ == "__main__":
    create_markers()
