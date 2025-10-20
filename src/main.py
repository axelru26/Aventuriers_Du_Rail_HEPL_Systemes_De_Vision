import os
from dotenv import load_dotenv

from media.video import process_video
from detection.assemble import assemble

load_dotenv()
URL = os.getenv("URL")


def main():
    process_video(URL, callback=assemble)

if __name__ == '__main__':
    main()
