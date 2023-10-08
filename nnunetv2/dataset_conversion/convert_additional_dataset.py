import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np 




# Set your directories
nifti_root_dir = '/home/j683r/local-work-temporary/lnq-data/additional_data/NIH'  # Root directory containing patient folders
output_dir = '/home/j683r/local-work-temporary/nnUNet_raw/Dataset919_only_additional_cut/'  # Output directory for modified NIfTI files
dicom_root_dir = '/home/j683r/local-work-temporary/lnq-data/additional_data/additional_nifti'  # Root directory containing patient folders
abdomen_lymph_dir = '/home/j683r/local-work-temporary/lnq-data/additional_data/MED_ABD_LYMPH_MASKS'
# Iterate through patient directories

labels = {
    "background": 0,
    "lymphnode": 1,
    "ignore": 2,
}
count = 1101
for patient_dir in os.listdir(nifti_root_dir):
    patient_dir_path = os.path.join(nifti_root_dir, patient_dir)

    # Skip if not a directory
    if not os.path.isdir(patient_dir_path):
        continue

    # Find NIfTI files in the patient directory
    nifti_files = [f for f in os.listdir(patient_dir_path) if f.endswith('.nii.gz')]

    # extract patient id
    patient_id = patient_dir.replace('Pat', '000')[-3:]  
    
    #match dicom folder name to pationt id
    dicom_dir = [f for f in os.listdir(dicom_root_dir) if (patient_id in f)]
    for dicom_dir in dicom_dir:
        if "MED" in dicom_dir:
        # Copy and rename the file
            dicom_img = [f for f in os.listdir(os.path.join(dicom_root_dir, dicom_dir)) if (".nii.gz" in f)][0]
            dicom_img_path = os.path.join(dicom_root_dir,dicom_dir, dicom_img)
            image = sitk.ReadImage(dicom_img_path)

            # Get the array data from the image
            nifti_data = sitk.GetArrayFromImage(image)

            # Create a new SimpleITK image from the modified data
            modified_image = sitk.GetImageFromArray(nifti_data)

            # Copy metadata and spacing information from the original image
            modified_image.SetDirection(image.GetDirection())
            modified_image.SetOrigin(image.GetOrigin())
            modified_image.SetSpacing(image.GetSpacing())

            # Save the modified image as a new NIfTI file
            sitk.WriteImage(modified_image, os.path.join(output_dir,"imagesTr", f"{count}_0000.nii.gz"))


            # Find the NIfTI file
            patient_id = patient_dir.replace('0', '').replace('Pat', 'pat')
            print(patient_id)
            nifti_filename = [f for f in nifti_files if patient_id in f][0]
            
            nifti_path = os.path.join(patient_dir_path, nifti_filename)
            # Load NIfTI data using SimpleITK
            nifti_img = sitk.ReadImage(nifti_path)
            # Get the NIfTI data as a numpy array
            nifti_array = sitk.GetArrayFromImage(nifti_img)
            nifti_array = nifti_array.astype(int)

            # set all values in the array to 1 (lymphnode) if they are not 0 (background)
            nifti_array[nifti_array != 0] = 1
            nifti_array= nifti_array[:, ::-1, :]
            nifti_array = nifti_array[slices_to_remove:, :, :]
            output_img = sitk.GetImageFromArray(nifti_array.astype(np.uint16))
            dicom_data = sitk.ReadImage(dicom_img_path)
            output_img.SetDirection(dicom_data.GetDirection())
            output_img.SetOrigin(dicom_data.GetOrigin())
            output_img.SetSpacing(dicom_data.GetSpacing())

            sitk.WriteImage(output_img, os.path.join(output_dir,"labelsTr", f"{count}.nii.gz"))
            
            print(output_img.GetMetaDataKeys())
            count += 1
            
