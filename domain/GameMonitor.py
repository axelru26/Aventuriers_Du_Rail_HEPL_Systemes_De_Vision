from enum import Enum, auto
import cv2

RED = (0, 0, 255)  # BGR


class GameState(Enum):
    """
    Represents the different states of the game monitoring process.
    """
    INITIALIZING = auto()
    CALIBRATING = auto()


class GameMonitor:
    """
    A state machine to monitor a Ticket To Ride game using computer vision.
    """

    def __init__(self, source):
        self.state = GameState.INITIALIZING
        self.source = source
        self.camera = None
        self.frame = None

    def run(self):
        """
        The main loop of the state machine.
        """
        if self.camera is None:
            self._initialize_camera()

        while True:
            self._get_frame()
            match self.state:
                case GameState.INITIALIZING:
                    raise ValueError(f"Camera should already be initialized")
                case GameState.CALIBRATING:
                    self._calibrate_camera()
                case _:
                    raise ValueError(f"Unknown state: {self.state}")

            cv2.imshow("Game Monitor", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        self._cleanup()

    def _initialize_camera(self):
        """
        Initializes the camera.
        """
        self.camera = cv2.VideoCapture(self.source)
        if not self.camera.isOpened():
            raise IOError("Could not open camera")
        self.state = GameState.CALIBRATING

    def _get_frame(self):
        """
        Gets the current frame from the camera.
        """
        ret, frame = self.camera.read()
        if not ret:
            raise IOError("Could not read frame")
        self.frame = frame

    def _cleanup(self):
        """
        Cleans up the camera and closes all windows.
        """
        self.camera.release()
        cv2.destroyAllWindows()

    def _calibrate_camera(self):
        cv2.putText(self.frame, f"STATE: {self.state.name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
