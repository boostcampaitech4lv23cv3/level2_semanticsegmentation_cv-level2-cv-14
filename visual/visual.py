import streamlit as st
import albumentations as A
import matplotlib.pyplot as plt
import torch

from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from visual_function import *


def load_data(coco_path="../data/train.json", data_mode="train"):
    coco = COCO(coco_path)
    transform = A.Compose([ToTensorV2()])
    dataset = CustomDataLoader(coco=coco, mode=data_mode, transform=transform)
    return dataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.write("Semantic Segmentation DATA EDA")
    col1, col2 = st.columns(2)
    with col1:
        coco_json = st.selectbox("json파일을 선택해주세요.", sorted(os.listdir("../data/")))
        coco_path = "../data/" + coco_json
        mode = st.radio("data mode", ("train", "test"))

        if (
            "coco_path" not in st.session_state
            or coco_path != st.session_state.coco_path
        ):
            st.session_state.dataset = load_data(coco_path, mode)
            st.session_state.coco_path = coco_path

    with col2:
        model_path = st.selectbox("model을 선택해주세요.", sorted(os.listdir("./saved/")))
        if (
            "model_path" not in st.session_state
            or model_path != st.session_state.model_path
        ):
            st.session_state.model = torch.load(
                "./saved/" + model_path, map_location=device
            )
            st.session_state.model_path = model_path

    if mode == "train":
        img_index = st.slider("이미지 Index", 0, len(st.session_state.dataset), 0, 1)
        temp_images, temp_masks, image_infos = st.session_state.dataset[img_index]
        st.write(image_infos["file_name"])
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 12))

        ax1.imshow(temp_images.permute([1, 2, 0]))
        ax1.grid(False)
        ax1.set_title("input image", fontsize=15)

        ax2.imshow(temp_masks)
        ax2.grid(False)
        ax2.set_title("masks", fontsize=15)

        pred = (
            st.session_state.model(temp_images.unsqueeze(0).to(device))
            .squeeze()
            .argmax(0)
            .cpu()
        )

        ax3.imshow(pred)
        ax3.grid(False)
        ax3.set_title("pred", fontsize=15)

        st.pyplot(fig)
        st.write("image shape:", temp_images.shape)
        st.write(
            "Unique values, category of transformed mask : \n",
            [{int(i), category_names[int(i)]} for i in list(np.unique(temp_masks))],
        )
    elif mode == "test":
        img_index = st.slider("이미지 Index", 0, len(st.session_state.dataset), 0, 1)
        temp_images, temp_masks, image_infos = st.session_state.dataset[img_index]
        st.write(image_infos["file_name"])
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))

        ax1.imshow(temp_images.permute([1, 2, 0]))
        ax1.grid(False)
        ax1.set_title("input image", fontsize=15)

        pred = (
            st.session_state.model(temp_images.unsqueeze(0).to(device))
            .squeeze()
            .argmax(0)
            .cpu()
        )

        ax2.imshow(pred)
        ax2.grid(False)
        ax2.set_title("pred", fontsize=15)

        st.pyplot(fig)
        st.write("image shape:", temp_images.shape)
        st.write(
            "Unique values, category of transformed mask : \n",
            [{int(i), category_names[int(i)]} for i in list(np.unique(temp_masks))],
        )


if __name__ == "__main__":
    main()
