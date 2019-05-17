import sys
import os
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).parent, ""))

import yaml
import tensorflow as tf 
import keras_yolo3.yolo as y

from models.base import BaseModel

CONFIG_FILE = os.path.join(Path(__file__).parent, 'config.yaml')
CONFIG_FILE_PERSON = os.path.join(Path(__file__).parent, 'config_person.yaml')


class Yolo3(BaseModel):
    def __init__(self):
        # init_path()
        with open(CONFIG_FILE, 'r') as ymlfile:
            config = yaml.load(ymlfile)
        if not config:
            raise Exception("config.yaml not found !")
      
        self.model = y.YOLO(model_path=os.path.join(Path(__file__).parent, config['model_path']),
                        anchors_path=os.path.join(Path(__file__).parent, config['anchors_path']),
                        classes_path=os.path.join(Path(__file__).parent, config['classes_path']),
                        font_path=os.path.join(Path(__file__).parent.parent, config['font_path']),
                        score=config['score_threshold'])

        self.graph = tf.get_default_graph()

    def predict(self, image_path=None):
        with self.graph.as_default():
            if image_path is not None:
                detected, detection_info = self.model.predict(image_path)
                list_bbox = []
                for box in detection_info['boxes']:
                    list_bbox.append(box['box'])

                return detected, list_bbox
            else:
                raise Exception("Image path cannot be None!")

    def predict_image(self, cv2_img=None):
        with self.graph.as_default():
            if cv2_img is not None:
                detected, detection_info = self.model.detect_person_cv2(cv2_img)
                list_bbox = []
                for box in detection_info['boxes']:
                    list_bbox.append(box['box'])

                return detected, list_bbox
            else:
                raise Exception("Image cannot be None!")
