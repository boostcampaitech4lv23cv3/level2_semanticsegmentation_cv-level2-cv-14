import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES


@PIPELINES.register_module()
class Gridmask(object):
    """
    Crop image by grid.

    Args:
        ratio (float): define grid mask size by ratio (grid length * ratio)
        'TODO'

    """
    def __init__(self, ratio, holes_number_x, holes_number_y, random_offset=False, p=1.0):
        self.ratio = ratio
        self.holes_num_x = holes_number_x
        self.holes_num_y = holes_number_y
        "TODO"
        self.random_offset = random_offset
        self.p = p

    def get_crop_bbox_list(self, img):
        """Generate grid and crop bounding box"""
        # x축 grid, y축 grid 당 각각의 길이 구함 
        grid_len_x = int(img.shape[1] / self.holes_num_x)
        grid_len_y = int(img.shape[0] / self.holes_num_y)
        
        "TODO"
        # random offset 
        offset = [0, 0]
        if self.random_offset:
            pass

        crop_bbox_list = []
        for i in range(self.holes_num_y):
            for j in range(self.holes_num_x):
                grid_x, grid_y = j * grid_len_x, i * grid_len_y
                crop_x1, crop_x2 = grid_x, grid_x + (grid_len_x * self.ratio)
                crop_y1, crop_y2 = grid_y, grid_y + (grid_len_y * self.ratio)
                
                crop_bbox_list.append([crop_y1, crop_y2, crop_x1, crop_x2])
        
        return crop_bbox_list

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):

        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline. 
            
        """
        img = results['img']
        crop_bbox_list = self.get_crop_bbox_list(img)

        # crop the image
        for crop_bbox in crop_bbox_list:
            img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            for crop_bbox in crop_bbox_list:
                results[key] = self.crop(results[key], crop_bbox)

        return results