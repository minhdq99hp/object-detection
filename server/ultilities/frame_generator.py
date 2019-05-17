import cv2
from contextlib import contextmanager
import os
from enum import Enum


@contextmanager
def video_capture(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    try:
        yield cap
    finally:
        cap.release()


class StreamMode(Enum):
    WEBCAM = -1
    IMAGE = 0
    IMAGE_DIR = 1
    VIDEO = 2
    RTSP = 3


class FrameGenerator:
    def __init__(self, mode, *args, **kwargs):
        self.mode = mode

        # SETUP IF IT IS VIDEO
        if self.mode == StreamMode.VIDEO:
            if len(args) > 0:
                self.path = args[0]

            vid = cv2.VideoCapture(self.path)
            self.vid_fps = vid.get(cv2.CAP_PROP_FPS)
            self.vid_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.vid_cc = vid.get(cv2.CAP_PROP_FOURCC)
            self.total_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)


            if not vid.isOpened():
                e = f"OpenCV can't open {self.path}"
                raise Exception(e)
            vid.release()

            self.cap = cv2.VideoCapture(self.path)

        elif self.mode == StreamMode.RTSP:
            if len(args) > 0:
                self.path = args[0]

            vid = cv2.VideoCapture(self.path)
            self.vid_fps = vid.get(cv2.CAP_PROP_FPS)
            self.vid_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.vid_cc = vid.get(cv2.CAP_PROP_FOURCC)

            if not vid.isOpened():
                e = f"OpenCV can't open {self.path}"
                raise Exception(e)
            vid.release()

            self.cap = cv2.VideoCapture(self.path)
            # CHECK WHETHER IF RTSP IS ALIVE
            # url = request.args['rtsp']
            #
            # code = urlopen(url).getcode()
            #
            # if str(code).startswith(('2', '3')):  # 2xx or 3xx are considered success
            #     print_header('RTSP IS WORKING')
            # else:
            #     print_header('RTSP IS DEAD')


    def yield_frame(self):
        if self.mode == StreamMode.WEBCAM:
            return self.yield_frame_from_webcam()
        elif self.mode == StreamMode.IMAGE:
            return self.yield_frame_from_image()
        elif self.mode == StreamMode.IMAGE_DIR:
            return self.yield_frame_from_image_dir()
        elif self.mode == StreamMode.VIDEO:
            return self.yield_frame_from_video()
        elif self.mode == StreamMode.RTSP:
            return self.yield_frame_from_rtsp()

    @staticmethod
    def yield_frame_from_webcam():
        with video_capture(0) as cap:
            while cap.isOpened():
                ret, img = cap.read()
                if ret:
                    yield img
                else:
                    break

    def yield_frame_from_image(self):
        yield cv2.imread(self.path)

    def yield_frame_from_video(self):
        with video_capture(self.path) as cap:
            while cap.isOpened():
                ret, img = cap.read()
                if ret:
                    yield img
                else:
                    break

    def yield_frame_from_rtsp(self):
        with video_capture(self.path) as cap:
            while cap.isOpened():
                ret, img = cap.read()
                if ret:
                    yield img
                else:
                    break

    def yield_frame_from_image_dir(self):
        for file in os.listdir(self.path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_path = self.path + file if self.path.endswith('/') else self.path + '/' + file
                yield cv2.imread(file_path)
