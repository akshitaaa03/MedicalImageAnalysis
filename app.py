import streamlit as st
import torch
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst,
    ScaleIntensity, ToTensor
)

from model import get_model
from utils import device
import nibabel as nib
import numpy as np


st.set_page_config(page_title="Medical Image Segmentation")
st.title("Brain Tumor Segmentation")

@st.cache_resource
def load_model():
    model = get_model().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

uploaded = st.file_uploader("Upload MRI (.nii / .nii.gz)")

'''transform = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    ToTensor()
])'''

from monai.transforms import (
    Compose, EnsureChannelFirst, ScaleIntensity, ToTensor
)

infer_transform = Compose([
    EnsureChannelFirst(),
    ScaleIntensity(),
    ToTensor()
])


if uploaded:
    with open("temp.nii.gz", "wb") as f:
        f.write(uploaded.read())

    #image = transform("temp.nii.gz").unsqueeze(0).to(device)
    nii = nib.load("temp.nii.gz")
    image_np = nii.get_fdata().astype("float32")

    image = infer_transform(image_np)
    image = image.unsqueeze(0).to(device)

    if st.button("Run Segmentation"):
        with torch.no_grad():
            pred = torch.argmax(model(image), dim=1)

        img = image.cpu().numpy()[0, 0]
        mask = pred.cpu().numpy()[0]
        s = img.shape[2] // 2

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img[:, :, s], cmap="gray")
        ax[0].set_title("MRI")
        ax[1].imshow(img[:, :, s], cmap="gray")
        ax[1].imshow(mask[:, :, s], cmap="hot", alpha=0.5)
        ax[1].set_title("Segmentation")
        st.pyplot(fig)
