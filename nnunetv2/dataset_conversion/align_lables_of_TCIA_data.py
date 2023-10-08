import os
import SimpleITK as sitk
import pydicom
import matplotlib.pyplot as plt
import numpy as np 
import dicom2nifti

# Set your directories
nifti_root_dir = '/home/j683r/local-work-temporary/lnq-data/additional_data/NIH'  # Root directory containing patient folders
output_dir = '/home/j683r/local-work-temporary/lnq-data/additional_data/additional_nifti_t'  # Output directory for modified NIfTI files
dicom_root_dir = '/home/j683r/local-work-temporary/lnq-data/additional_data/manifest-1680277513580/CT Lymph Nodes'  # Root directory containing patient folders
# Iterate through patient directories

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
        dicom_dir_path = os.path.join(dicom_root_dir, dicom_dir)
    
        dicom_dir = [f for f in os.listdir(dicom_dir_path)][0]
        dicom_dir_path = os.path.join(dicom_dir_path, dicom_dir)
        
        try:
            dicom_dir = [f for f in os.listdir(dicom_dir_path) if ("lymph" in f)][0]
            dicom_dir_path = os.path.join(dicom_dir_path, dicom_dir)
        except IndexError:
            print(f"no folder for {patient_id}")
            continue
        dicom_image = os.path.join(dicom_dir_path, [f for f in os.listdir(dicom_dir_path) if (".dcm" in f)][0])

        # #load dicom data
        # dicom_ds = pydicom.dcmread(dicom_image)
        # #save as nifti
        # dicom_img = sitk.ReadImage(dicom_image)
        
        #check if path contains ABD
        if "ABD" in dicom_dir_path:
            output_dir_tmp = os.path.join(output_dir, f"ABD_{patient_id}")
            if not os.path.exists(output_dir_tmp):
                os.mkdir(output_dir_tmp)
            print(dicom_dir_path)
            bashCommand = f"dcm2niix -o {output_dir_tmp} \"{dicom_dir_path}\""
            os.system(bashCommand)
            output_dir_tmp = output_dir
        #check if path contains MED
        elif "MED" in dicom_dir_path:
            output_dir_tmp = os.path.join(output_dir, f"MED_{patient_id}")
            if not os.path.exists(output_dir_tmp):
                os.mkdir(output_dir_tmp)
            print(f"dicom_dir_path: {dicom_dir_path}")
            bashCommand = f"dcm2niix -o {output_dir_tmp} \"{dicom_dir_path}\""
            os.system(bashCommand)
            output_dir_tmp = output_dir
        else:
            output_dir_tmp = os.path.join(output_dir, f"{patient_id}")
            if not os.path.exists(output_dir_tmp):
                os.mkdir(output_dir_tmp)
            dicom2nifti.convert_directory(dicom_dir_path, output_dir_tmp)
            print(f"no folder for {patient_id}")
            output_dir_tmp = output_dir

        # # Process each NIfTI file for the current patient
        # for nifti_filename in nifti_files:
        #     nifti_path = os.path.join(patient_dir_path, nifti_filename)

        #     # Load NIfTI data using SimpleITK
        #     nifti_img = sitk.ReadImage(nifti_path)

        #     dicom_ds = pydicom.dcmread(dicom_image)

        #     # Get DICOM  z-coordinate
        #     ds_origin = dicom_ds.ImagePositionPatient

        #     #Get Dicom spacing
        #     ds_spacing = dicom_ds.PixelSpacing

        #     # Calculate the z-coordinate for each slice in the NIfTI volume
        #     n_slices = nifti_img.GetDepth()
            
        #     #print information
        #     print(f"Patient ID: {patient_id}")
        #     print(f"Number of slices: {n_slices} nifti")
        #     print(f"Spacing: {ds_spacing} dicom, {nifti_img.GetSpacing()} nifti")
        #     print(f"Origin: {ds_origin} dicom, {nifti_img.GetOrigin() } nifti")

        #     # Update the NIfTI metadata
        #     # nifti_img.SetSpacing((ds_spacing[0],ds_spacing[1],nifti_img.GetSpacing()[2]))
        #     nifti_img.SetOrigin((ds_origin[0], ds_origin[1], ds_origin[2]))

        #     # Save the modified NIfTI file
        #     modified_nifti_filename = f"modified_{nifti_filename}"
        #     modified_nifti_path = os.path.join(output_dir, modified_nifti_filename)

        #     #after conversion
        #     print("After conversion")
        #     print(f"Patient ID: {patient_id}")
        #     print(f"Number of slices: {n_slices} nifti")
        #     print(f"Spacing: {ds_spacing} dicom, {nifti_img.GetSpacing()} nifti")
        #     print(f"Origin: {ds_origin} dicom, {nifti_img.GetOrigin() } nifti")
        #     sitk.WriteImage(nifti_img, modified_nifti_path)

        #     print(f"Modified NIfTI file saved: {modified_nifti_path}")
        #     exit()
