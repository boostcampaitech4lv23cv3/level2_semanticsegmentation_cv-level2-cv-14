import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
import albumentations
from albumentations import Compose

from ..builder import PIPELINES


@PIPELINES.register_module()
class Gridmask(object):
    """
    Cutout image & seg by grid.

    'TODO'
    Args:
        ratio (float): define grid mask size by ratio (grid length * ratio)
        holes_number_x (int): 
        holes_number_y (int):
        random_offset (float):
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
        seg_fill_in (int): The labels of pixel to fill in the dropped regions.
            If seg_fill_in is None, skip. Default: None.
        prob (float): gridmask probability
    """
    def __init__(self, 
                 ratio, 
                 holes_number_x, 
                 holes_number_y, 
                 random_offset=False,
                 fill_in=(0, 0, 0),
                 seg_fill_in=None,
                 prob=1.0):

        assert 0 < ratio < 1
        assert holes_number_x > 0 and holes_number_y > 0
        assert 0 <= prob <= 1
        if seg_fill_in is not None:
            assert (isinstance(seg_fill_in, int) and 0 <= seg_fill_in
                    and seg_fill_in <= 255)

        self.ratio = ratio
        self.holes_num_x = holes_number_x
        self.holes_num_y = holes_number_y
        self.random_offset = random_offset #"TODO"
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.prob = prob

    def get_grid_len(self, img):
        """get cutout box w, h"""
        grid_w = int(img.shape[1] / self.holes_num_x)
        grid_h = int(img.shape[0] / self.holes_num_y)
        return grid_w, grid_h

    def get_cutout_grid_list(self, grid_w, grid_h):
        """Generate grid coordinate list."""
        offset_x, offset_y = 0, 0
        if self.random_offset:
            offset_x = np.random.randint(0, grid_w * (1-self.ratio))
            offset_y = np.random.randint(0, grid_h * (1-self.ratio))

        cutout_grid_list = []
        for i in range(self.holes_num_y):
            for j in range(self.holes_num_x):
                grid_x, grid_y = (j * grid_w) + offset_x, (i * grid_h) + offset_y
                cutout_grid_list.append([grid_x, grid_y])
        
        return cutout_grid_list

    def __call__(self, results):
        """Call function to drop regions of image."""
        cutout = True if np.random.rand() < self.prob else False

        if cutout:
            img = results['img']
            h, w, c = img.shape
            grid_w, grid_h = self.get_grid_len(img)
            n_holes = self.get_cutout_grid_list(grid_w, grid_h)
            cutout_w, cutout_h = int(grid_w * self.ratio), int(grid_h * self.ratio)

            for hole in n_holes:
                x1, y1 = hole[0], hole[1]

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                results['img'][y1:y2, x1:x2, :] = self.fill_in

                if self.seg_fill_in is not None:
                    for key in results.get('seg_fields', []):
                        results[key][y1:y2, x1:x2] = self.seg_fill_in

        return results


# https://github.com/open-mmlab/mmsegmentation/pull/392 
@PIPELINES.register_module()
class Albu(object):
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        keymap (dict): Contains {'input key':'albumentation-style key'}
    """

    def __init__(self,
                 transforms,
                #  bbox_params=None,
                 keymap=None,
                 update_pad_shape=False):

        transforms = copy.deepcopy(transforms)

        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape

        self.aug = Compose([self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_semantic_seg': 'masks',
                # 'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        It inherits some of :func:`build_from_cfg` logic.
        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.
        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        results = self.aug(**results)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str