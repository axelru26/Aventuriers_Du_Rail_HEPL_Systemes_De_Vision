from enum import Enum, auto
import cv2
import numpy as np

ARUCO_DICT_NAME = "DICT_4X4_50"
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()
ARUCO_VALID_IDS = {0, 1, 2, 3}

BOARD_SIZE = (1200, 800)


class GameState(Enum):
    """
    Represents the different states of the game monitoring process.
    """
    INITIALIZING = auto()
    CALIBRATING_WARP = auto()
    CALIBRATING_COLORS = auto()


class GameMonitor:
    """
    A state machine to monitor a Ticket To Ride game using computer vision.
    """

    def __init__(self, source):
        self.state = GameState.INITIALIZING
        self.source = source
        self.camera = None
        self.frame = None
        self.warped_frame = None
        self.current_window_name = None

    def run(self):
        """
        The main loop of the state machine.
        """
        if self.camera is None:
            self._initialize_camera()
            self.state = GameState.CALIBRATING_WARP

        while True:
            self._get_frame()
            match self.state:
                case GameState.CALIBRATING_WARP:
                    self._calibrate_warp()
                    self._show_frame("Calibrating Warp", self.frame)
                case GameState.CALIBRATING_COLORS:
                    self.warped_frame = cv2.warpPerspective(self.frame, self.perspective_matrix, BOARD_SIZE)
                    self._show_frame("Calibrating Colors", self.warped_frame)
                case _:
                    raise ValueError(f"Unknown state: {self.state}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # press 'q' to quit
                break
            elif key == ord('r'):  # Press 'r' to return to warp calibration
                self.state = GameState.CALIBRATING_WARP

        self._cleanup()

    # RUN METHODS
    def _get_frame(self):
        """
        Reads a frame from the camera.
        """
        ret, frame = self.camera.read()
        if not ret:
            raise IOError("Could not read frame")
        self.frame = frame

    def _cleanup(self):
        """
        Cleans up the camera and any open windows.
        """
        self.camera.release()
        cv2.destroyAllWindows()
        self.current_window_name = None

    # INITIALIZATION METHODS
    def _initialize_camera(self):
        """
        Initializes the camera.
        """
        self.camera = cv2.VideoCapture(self.source)
        if not self.camera.isOpened():
            raise IOError("Could not open camera")

    def _show_frame(self, window_name, frame):
        """
        A helper to display a frame in a window.
        If the window name changes, it closes the old window and opens a new one.
        """
        if self.current_window_name != window_name:
            if self.current_window_name:
                cv2.destroyWindow(self.current_window_name)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, BOARD_SIZE[0], BOARD_SIZE[1])
            self.current_window_name = window_name

        cv2.imshow(window_name, frame)

    # WARP CALIBRATION METHODS
    def _calibrate_warp(self):
        """
        Handles the warp calibration process.
        """
        # Initialize or cleanup variables
        self.aruco_ids = None
        self.aruco_corners = None
        self.perspective_matrix = None

        self._detect_aruco()
        if self.aruco_corners is not None and self.aruco_ids is not None:
            cv2.putText(self.frame, f"Found {len(self.aruco_ids)} ArUco markers", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.aruco.drawDetectedMarkers(self.frame, self.aruco_corners, self.aruco_ids)
            if len(self.aruco_ids) == 4:
                self._get_perspective_matrix()
                self.state = GameState.CALIBRATING_COLORS
            return
        cv2.putText(self.frame, "No ArUco markers found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _detect_aruco(self):
        """
        Detects ArUco markers in the current frame.
        """
        gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)
        corners, ids, rejected = detector.detectMarkers(gray_frame)
        if ids is not None:
            self._filter_aruco_ids(corners, ids)

    def _filter_aruco_ids(self, corners, ids):
        """
        Keep only ArUco markers with valid IDs.
        """
        filtered_corners = []
        filtered_ids = []
        for i, corner in zip(ids.flatten(), corners):
            if i in ARUCO_VALID_IDS:
                filtered_ids.append([i])
                filtered_corners.append(corner)
        self.aruco_ids = np.array(filtered_ids, dtype=np.int32)
        self.aruco_corners = tuple(filtered_corners)

    def _get_perspective_matrix(self):
        """
        Calculates the perspective transformation matrix based on the detected
        ArUco markers. It assumes ARUCO_VALID_IDS (0, 1, 2, 3) correspond to
        the physical Top-Left, Top-Right, Bottom-Right, and Bottom-Left
        corners of the board, respectively.
        """
        marker_corners_by_id = {}
        for i, corner_data in zip(self.aruco_ids.flatten(), self.aruco_corners):
            marker_corners_by_id[i] = corner_data[0]
        src_points = np.float32([
            marker_corners_by_id[0][0],
            marker_corners_by_id[1][0],
            marker_corners_by_id[2][0],
            marker_corners_by_id[3][0]
        ])
        dst_points = np.float32([
            [0, 0],
            [BOARD_SIZE[0] - 1, 0],
            [BOARD_SIZE[0] - 1, BOARD_SIZE[1] - 1],
            [0, BOARD_SIZE[1] - 1]
        ])
        self.perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # COLOR CALIBRATION METHODS
