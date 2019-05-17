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
            nn_cfg = yaml.load(ymlfile)
        if not nn_cfg:
            raise ("config.yaml not found !")
      
        self.model = y.YOLO(model_path=os.path.join(Path(__file__).parent, nn_cfg['model_path']),
                        anchors_path=os.path.join(Path(__file__).parent, nn_cfg['anchors_path']),
                        classes_path=os.path.join(Path(__file__).parent, nn_cfg['classes_path']),
                        font_path=os.path.join(Path(__file__).parent.parent, nn_cfg['font_path']),
                        score=nn_cfg['score_threshold'])

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
                raise ("Image path cannot be None!")

    def predict_image(self, cv2_img=None):
        with self.graph.as_default():
            if cv2_img is not None:
                detected, detection_info = self.model.detect_person_cv2(cv2_img)
                list_bbox = []
                for box in detection_info['boxes']:
                    list_bbox.append(box['box'])

                return detected, list_bbox
            else:
                raise ("Image cannot be None!")

class Yolo3_Person(BaseModel):
    def __init__(self):
        # init_path()
        with open(CONFIG_FILE_PERSON, 'r') as ymlfile:
            nn_cfg = yaml.load(ymlfile)
        if not nn_cfg:
            raise ("config_person.yaml not found !")
      
        self.model = y.YOLO(model_path=os.path.join(Path(__file__).parent, nn_cfg['model_path']),
                        anchors_path=os.path.join(Path(__file__).parent, nn_cfg['anchors_path']),
                        classes_path=os.path.join(Path(__file__).parent, nn_cfg['classes_path']),
                        font_path=os.path.join(Path(__file__).parent.parent, nn_cfg['font_path']),
                        score=nn_cfg['score_threshold'])

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
                raise ("Image path cannot be None!")

    def predict_image(self, cv2_img=None):
        with self.graph.as_default():
            if cv2_img is not None:
                detected, detection_info = self.model.detect_person_cv2(cv2_img)
                list_bbox = []
                for box in detection_info['boxes']:
                    list_bbox.append(box['box'])

                return detected, list_bbox
            else:
                raise ("Image cannot be None!")
