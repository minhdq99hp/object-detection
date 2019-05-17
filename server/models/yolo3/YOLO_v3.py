import sys

import os
from os.path import realpath, dirname

sys.path.append(dirname(realpath(__file__)))

import yaml
import tensorflow as tf

import keras_yolo3.yolo as y

from cv2 import imread

from models.base import BaseModel

CONFIG_FILE = os.path.join(dirname(realpath(__file__)), 'config.yaml')


class Yolo3(BaseModel):
    def __init__(self):
        # init_path()
        with open(CONFIG_FILE, 'r') as ymlfile:
            config = yaml.load(ymlfile)
        if not config:
            raise Exception("config.yaml not found !")
      
        self.model = y.YOLO(model_path=os.path.join(dirname(realpath(__file__)), config['model_path']),
                            anchors_path=os.path.join(dirname(realpath(__file__)), config['anchors_path']),
                            classes_path=os.path.join(dirname(realpath(__file__)), config['classes_path']),
                            font_path='/home/minhdq99hp/object-detection/server/models/yolo3/keras_yolo3/font/FiraMono-Medium.otf',
                            score=config['score_threshold'])

        self.graph = tf.get_default_graph()

    def predict(self, image_path=None):
        with self.graph.as_default():
            if image_path is not None:
                cv2_img = imread(image_path)
                detected, detection_info = self.model.detect_image(cv2_img)
                list_bbox = []
                for box in detection_info['boxes']:
                    list_bbox.append(box['box'])

                return detected, list_bbox
            else:
                raise Exception("Image path cannot be None!")

    def predict_image(self, cv2_img=None):
        with self.graph.as_default():
            if cv2_img is not None:
                return self.model.predict_image(cv2_img)
            else:
                raise Exception("Image cannot be None!")
