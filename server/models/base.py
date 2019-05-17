import pickle
import os
import numpy as np
import sys

from abc import ABC, abstractmethod
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).parent, ""))


class BaseModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(image_path=None):
        pass
    
    @abstractmethod
    def predict_image(cv2_image=None):
    	pass 

    def evaluate(self, data_path, annots_path):
        '''
        annotations file: dictionary {im1_path: count_1, im2_path: count_2, ...}
        ** Image path is relative to data_path.
        For example: 
        - data_path: '/home/yoshi/Downloads/infore_data_1'
        - im1_path: '6_01_R_032019160000/88.jpg'
        -> full path of 1st image: '/home/yoshi/Downloads/infore_data_1/6_01_R_032019160000/88.jpg'
        '''

        annotations = pickle.load(open(annots_path, 'rb'))
        predictions = {}
        for i, (im_path, count) in enumerate(annotations.items()):
            print("Evaluating {}/{} ... ".format(i + 1, len(annotations)), end='\r', flush=True)
            full_im_path = os.path.join(data_path, im_path)
            detected, list_bbox = self.predict(full_im_path)
            pred_count = len(list_bbox)
            predictions[im_path] = pred_count

        all_annots = list(annotations.values())
        all_preds = list(predictions.values())

        assert len(all_preds) == len(all_annots), "Number of images and annots mismatch!" 
        a = np.array(all_preds)
        b = np.array(all_annots)
        mean_dev = float(sum(abs(a-b)))/len(all_preds)
        mean_dev_per = float(sum(abs((a-b)/b)))/len(all_preds)
        correct = np.sum(a == b) / len(all_preds)
        return mean_dev, mean_dev_per, correct