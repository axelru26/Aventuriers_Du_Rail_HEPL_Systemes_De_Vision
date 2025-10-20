import numpy as np
import cv2

from .board import detect_board, warped_board

def assemble(frame):
    output_or_transformation_matrix, _, _, detect = detect_board(frame)
    if detect:
        board_warped = warped_board(frame)
        return board_warped

    return output_or_transformation_matrix