import cv2
import numpy as np

WIDTH, HEIGHT = 1200, 800
DESTINATIONS = np.array([
    [0, 0],
    [WIDTH - 1, 0],
    [WIDTH - 1, HEIGHT - 1],
    [0, HEIGHT - 1]
], dtype="float32")
IDENTITY = np.eye(3, dtype=np.float32)

def detect_board(frame: np.ndarray):
    detect = False
    output_frame = frame.copy()

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        cv2.putText(output_frame, "4 ArUco requis", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return output_frame, WIDTH, HEIGHT, detect

    top_left_points = {}
    for i, corner in enumerate(corners):
        marker_id = int(ids[i][0])
        top_left = tuple(corner[0][0])  # coin haut gauche
        top_left_points[marker_id] = top_left

    try:
        # 0 = haut gauche, 1 = haut droit, 2 = bas droit, 3 = bas gauche
        points_source = np.array([
            top_left_points[3],
            top_left_points[4],
            top_left_points[6],
            top_left_points[5]
        ], dtype="float32")
    except KeyError:
        cv2.putText(output_frame, "Mauvais IDs detectes", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return output_frame, WIDTH, HEIGHT, detect

    cv2.polylines(output_frame, [np.int32(points_source)], isClosed=True, color=(0, 255, 255), thickness=3)

    transformation_matrix = cv2.getPerspectiveTransform(points_source, DESTINATIONS)

    return transformation_matrix, WIDTH, HEIGHT, detect

def warped_board(frame: np.ndarray):
    matrix, width, height,_ = detect_board(frame)
    board_warped = cv2.warpPerspective(frame, matrix, (width, height))
    return board_warped