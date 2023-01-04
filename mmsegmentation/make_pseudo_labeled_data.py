import os
import os.path as osp
import shutil
import numpy as np
import torch
import tqdm
import pickle
import cv2
import matplotlib.pyplot as plt


def make_pseudo(config, teacher_checkpoint):
    """make_pseudo

    Args:
        config (str): config 파일의 경로를 입력해주세요
        teacher_checkpoint (str): 학습된 파일의 경로를 입력해주세요
    """
    # argparse
    cnt = 1
    out_path = osp.join("/".join(teacher_checkpoint.split("/")[:-1]), "out.pickle")

    # 폴더 생성하기
    src_dir = "/opt/ml/input/data/data_new"
    dst_dir = f"/opt/ml/input/data/pseudo_labeled_{cnt}"
    if not osp.exists(dst_dir):
        os.mkdir(dst_dir)
        os.mkdir(dst_dir + "/images")
        os.mkdir(dst_dir + "/images/train")
        os.mkdir(dst_dir + "/images/val")
        os.mkdir(dst_dir + "/images/test")
        os.mkdir(dst_dir + "/annotations")
        os.mkdir(dst_dir + "/annotations/train")
        os.mkdir(dst_dir + "/annotations/val")

    # data 복사
    ## train
    filenames = os.listdir(osp.join(src_dir, "images", "train"))
    for file in tqdm.tqdm(filenames):
        src = osp.join(src_dir, "images", "train", file)
        dst = osp.join(dst_dir, "images", "train", file)
        shutil.copy(src, dst)
    ## valid
    filenames = os.listdir(osp.join(src_dir, "images", "val"))
    for file in tqdm.tqdm(filenames):
        src = osp.join(src_dir, "images", "val", file)
        dst = osp.join(
            dst_dir, "images", "val", file
        )  # validation set도 training에 포함시키려면 해당 부분 "train"으로 수정해야함
        shutil.copy(src, dst)
    ## test
    filenames = os.listdir(osp.join(src_dir, "images", "test"))
    for file in tqdm.tqdm(filenames):
        src = osp.join(src_dir, "images", "test", file)
        dst = osp.join(dst_dir, "images", "train", file)
        shutil.copy(src, dst)

    # test data 복사
    filenames = os.listdir(osp.join(src_dir, "images", "test"))
    for file in tqdm.tqdm(filenames):
        src = osp.join(src_dir, "images", "test", file)
        dst = osp.join(dst_dir, "images", "test", file)
        shutil.copy(src, dst)

    # annotation 복사
    ## train
    filenames = os.listdir(osp.join(src_dir, "annotations", "train"))
    for file in tqdm.tqdm(filenames):
        src = osp.join(src_dir, "annotations", "train", file)
        dst = osp.join(dst_dir, "annotations", "train", file)
        shutil.copy(src, dst)
    ## val
    filenames = os.listdir(osp.join(src_dir, "annotations", "val"))
    for file in tqdm.tqdm(filenames):
        src = osp.join(src_dir, "annotations", "val", file)
        dst = osp.join(
            dst_dir, "annotations", "val", file
        )  # validation set도 training에 포함시키려면 해당 부분 "train"으로 수정해야함
        shutil.copy(src, dst)

    dst = osp.join(dst_dir, "annotations", "train")

    # test annotation 생성
    os.system(
        f"python /opt/ml/input/code/mmsegmentation/tools/model_ensemble.py --config {config} --checkpoint {teacher_checkpoint} --aug-test --out {dst} --gpus 0"
    )

    print("mmsegmentation의 dataset config 경로를 변경 해주세요!")


if __name__ == "__main__":
    config = "/opt/ml/input/code/mmsegmentation/work_dirs/knet_uper/knet_uper.py"
    teacher_checkpoint = (
        "/opt/ml/input/code/mmsegmentation/work_dirs/knet_uper/best_mIoU_iter_80000.pth"
    )
    make_pseudo(config, teacher_checkpoint)
