import os
import nibabel as nib
import numpy as np
import nrrd
import SimpleITK as sitk
from bpreg.scripts.bpreg_inference import bpreg_inference

base = r"/home/j683r/local-work-temporary/lnq-data/raw_data/Dataset911_LNQ"

if not os.path.exists(base):
    print ("path missing")
    exit()

# setup paths
result_path = r'/home/j683r/local-work-temporary/lnq-data/nifti'
training_path = os.path.join(base,'imagesTr' ) 
testing_path = os.path.join(base,'imagesTs' )
result_training_path = os.path.join(result_path,'imagesTr' )
result_testing_path = os.path.join(result_path,'imagesTs' )
inference_path = r'/home/j683r/local-work-temporary/bpreg'
inference_training_path = os.path.join(inference_path,'imagesTr' )
inference_testing_path = os.path.join(inference_path,'imagesTs' )



#maybe create directories
if not os.path.exists(result_path):
    os.makedirs(result_path)

if not os.path.exists(result_training_path):
    os.makedirs(result_training_path)

if not os.path.exists(result_testing_path):
    os.makedirs(result_testing_path)


# convert testing data from nrrd to nifti
for file in os.listdir(testing_path):
    #check if file already exists and skip if it does
    if os.path.exists(os.path.join(result_testing_path, file.replace(".nrrd", ".nii.gz"))):
        print("skipping: ", file)
        continue

    if file.endswith(".nrrd"):
        print("converting: ", file)


    # Read the NRRD image using SimpleITK
    nrrd_image = sitk.ReadImage(os.path.join(testing_path, file))

    nifti_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(nrrd_image))
    nifti_image.SetOrigin(nrrd_image.GetOrigin())         # Preserve origin
    nifti_image.SetSpacing(nrrd_image.GetSpacing())       # Preserve voxel spacing
    nifti_image.SetDirection(nrrd_image.GetDirection())   # Preserve direction matrix


    sitk.WriteImage(nifti_image, os.path.join(result_testing_path, file.replace(".nrrd", ".nii.gz")))


# convert training data from nrrd to nifti
for file in os.listdir(training_path):
    #check if file already exists and skip if it does
    if os.path.exists(os.path.join(result_training_path, file.replace(".nrrd", ".nii.gz"))):
        print("skipping: ", file)
        continue

    if file.endswith(".nrrd"):
        print("converting: ", file)


    # Read the NRRD image using SimpleITK
    nrrd_image = sitk.ReadImage(os.path.join(training_path, file))

    nifti_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(nrrd_image))
    nifti_image.SetOrigin(nrrd_image.GetOrigin())         # Preserve origin
    nifti_image.SetSpacing(nrrd_image.GetSpacing())       # Preserve voxel spacing
    nifti_image.SetDirection(nrrd_image.GetDirection())   # Preserve direction matrix


    sitk.WriteImage(nifti_image, os.path.join(result_training_path, file.replace(".nrrd", ".nii.gz")))

bpreg_inference(result_testing_path,inference_testing_path)
bpreg_inference(result_training_path,inference_training_path)
