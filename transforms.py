'''from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, RandCropByPosNegLabeld, ToTensord
)

def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64,64,64),
            pos=1, neg=1, num_samples=2
        ),
        ToTensord(keys=["image", "label"])
    ])'''
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    EnsureType
)

transform = Compose([
    LoadImage(image_only=True),   # 🔑 VERY IMPORTANT
    EnsureChannelFirst(),
    ScaleIntensity(),
    EnsureType()
])

