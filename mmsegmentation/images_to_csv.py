import cv2
import numpy as np
import pandas as pd
import os.path as osp
import os
import tqdm
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser(description="soft ensemble output to csv")
parser.add_argument("ENSEMBLE_DIR", help="path of soft-ensemble output")
args = parser.parse_args()

# user define functions
size = 256
to_tensor = A.Compose([ToTensorV2()])
resize = A.Compose([A.Resize(size, size)])


def restore_file_name(file_name):
    if "test_1" in file_name:
        return "batch_01_vt/" + file_name[6:11] + "jpg"
    if "test_2" in file_name:
        return "batch_02_vt/" + file_name[6:11] + "jpg"
    if "test_3" in file_name:
        return "batch_03/" + file_name[6:11] + "jpg"


def image_read(file_path):
    images = cv2.imread(file_path)
    # images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
    # images /= 255.0
    return images


# main

# read file names (*.png)
file_names = [img for img in os.listdir(args.ENSEMBLE_DIR) if img.endswith(".png")]

# read soft-ensemble output
imgs = [
    resize(image=image_read(args.ENSEMBLE_DIR + img))["image"][:, :, 0]
    for img in file_names
]

# modify file names
corrected_file_names = [restore_file_name(img) for img in file_names]

# reshape
preds_array = np.empty((0, size * size), dtype=np.long)
oms = np.array(imgs)
oms = oms.reshape([oms.shape[0], size * size]).astype(int)
preds = np.vstack((preds_array, oms))

# sample_submisson.csv 열기
submission = pd.read_csv(
    "/opt/ml/input/code/submission/sample_submission.csv", index_col=None
)

# PredictionString 대입
for file_name, string in tqdm.tqdm(zip(corrected_file_names, preds)):
    submission = submission.append(
        {
            "image_id": file_name,
            "PredictionString": " ".join(str(e) for e in string.tolist()),
        },
        ignore_index=True,
    )

# submission.csv로 저장
submission.to_csv(args.ENSEMBLE_DIR + "submission.csv", index=False)
