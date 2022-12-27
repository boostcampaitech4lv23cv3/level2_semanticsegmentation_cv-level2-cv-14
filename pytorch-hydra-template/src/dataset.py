from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pycocotools.coco import COCO
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import os, sys
from seed_everything import seedEverything, _init_fn

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from copypaste import copy_paste

# seed
seedEverything(42)

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


class CustomDataSet(Dataset):
    """COCO format"""

    # mode: train, val, test

    def __init__(self, data_dir, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        self.coco = COCO(data_dir + "/" + mode + ".json")
        self.category_names = [
            "Backgroud",
            "General trash",
            "Paper",
            "Paper pack",
            "Metal",
            "Glass",
            "Plastic",
            "Styrofoam",
            "Plastic bag",
            "Battery",
            "Clothing",
        ]

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_dir, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0
        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)
            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx: idx["area"], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                pixel_value = self.category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos

        if self.mode == "test":
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


@copy_paste.copy_paste_class
class CopyPasteDataSet(Dataset):
    """COCO format"""

    # mode: train, val, test

    def __init__(self, data_dir, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.data_dir = data_dir
        self.coco = COCO(data_dir + "/" + mode + ".json")
        self.category_names = [
            "Backgroud",
            "General trash",
            "Paper",
            "Paper pack",
            "Metal",
            "Glass",
            "Plastic",
            "Styrofoam",
            "Plastic bag",
            "Battery",
            "Clothing",
        ]

    def load_example(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.data_dir, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)
            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)
            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0

            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx: idx["area"], reverse=True)
            bboxes, masks = [], []
            for i in range(len(anns)):
                category_id = anns[i]["category_id"]
                className = get_classname(category_id, cats)
                pixel_value = self.category_names.index(className)
                x, y, x2, y2 = anns[i]["bbox"]
                bboxes.append([x, y, x2, y2, int(category_id), int(i)])
                mask = np.zeros((image_infos["height"], image_infos["width"]))
                mask[self.coco.annToMask(anns[i]) == 1] = pixel_value
                mask.astype(np.int8)
                masks.append(mask)

            return {"image": images, "masks": masks, "bboxes": bboxes}, image_infos

        if self.mode == "test":
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
