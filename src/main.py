import os
from dotenv import load_dotenv

from media.video import process_video
from detection.board import detect_board

load_dotenv()
URL = os.getenv("URL")


def main():
    process_video(URL, callback=detect_board)


if __name__ == '__main__':
    main()
