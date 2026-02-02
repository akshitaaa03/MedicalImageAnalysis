import numpy as np
import nibabel as nib
import os

os.makedirs("data/imagesTr", exist_ok=True)
os.makedirs("data/labelsTr", exist_ok=True)

image = np.random.rand(64, 64, 64)

label = np.zeros((64, 64, 64))
label[20:40, 20:40, 20:40] = 1

nib.save(nib.Nifti1Image(image, np.eye(4)), "data/imagesTr/sample.nii.gz")
nib.save(nib.Nifti1Image(label, np.eye(4)), "data/labelsTr/sample_seg.nii.gz")

print("Dummy dataset created")
