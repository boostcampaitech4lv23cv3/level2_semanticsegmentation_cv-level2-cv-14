import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES


@PIPELINES.register_module()
class Gridmask(object):
    """
    Cutout image & seg by grid.

    Args:
        ratio (float): define grid mask size by ratio (grid length * ratio)
        p (float): gridmast probability
        'TODO'
    """
    def __init__(self, 
                 ratio, 
                 holes_number_x, 
                 holes_number_y, 
                 random_offset=False,
                 fill_in=(0, 0, 0),
                 seg_fill_in=None,
                 p=1.0):

        assert 0 < ratio < 1
        assert holes_number_x > 0 and holes_number_y > 0
        assert 0 <= p <= 1
        if seg_fill_in is not None:
            assert (isinstance(seg_fill_in, int) and 0 <= seg_fill_in
                    and seg_fill_in <= 255)

        self.ratio = ratio
        self.holes_num_x = holes_number_x
        self.holes_num_y = holes_number_y
        self.random_offset = random_offset #"TODO"
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.p = p

    def get_grid_len(self, img):
        """get cutout box w, h"""
        grid_w = int(img.shape[1] / self.holes_num_x)
        grid_h = int(img.shape[0] / self.holes_num_y)
        return grid_w, grid_h

    def get_cutout_grid_list(self, grid_w, grid_h):
        """Generate grid coordinate list."""
        # random offset "TODO"
        offset = [0, 0]
        if self.random_offset:
            pass

        cutout_grid_list = []
        for i in range(self.holes_num_y):
            for j in range(self.holes_num_x):
                grid_x, grid_y = (j * grid_w) + offset[0], (i * grid_h) + offset[1]
                cutout_grid_list.append([grid_x, grid_y])
        
        return cutout_grid_list

    def __call__(self, results):
        """Call function to drop regions of image."""
        cutout = True if np.random.rand() < self.p else False

        if cutout:
            img = results['img']
            h, w, c = img.shape
            grid_w, grid_h = self.get_grid_len(img)
            n_holes = self.get_cutout_grid_list(grid_w, grid_h)
            cutout_w, cutout_h = int(grid_w * self.ratio), int(grid_h * self.ratio)

            for hole in range(n_holes):
                x1, y1 = hole[0], hole[1]

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                results['img'][y1:y2, x1:x2, :] = self.fill_in

                if self.seg_fill_in is not None:
                    for key in results.get('seg_fields', []):
                        results[key][y1:y2, x1:x2] = self.seg_fill_in

        return results