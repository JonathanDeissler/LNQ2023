from pathlib import Path
import os
import shutil
import sys
import time
from typing import Sequence
import nrrd
import numpy as np
import scipy
from tqdm import tqdm
import vtk
import SimpleITK as sitk
from scipy.ndimage import morphology
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json
import torch
from skimage import morphology as sk_morphology
import json


from totalsegmentator.python_api import totalsegmentator


def convert_types(nrrd_file: str, nifti_file: str):
    """Convert nrrd file to nifti file."""
    im = sitk.ReadImage(nrrd_file)
    sitk.WriteImage(im, nifti_file)


def convert(input_image: list[Path], output_path: Path, append_index_suffix=False):
    """Convert all nrrd files in input_images to nifti files in output_path."""
    output_path.mkdir(exist_ok=True, parents=True)
    for im in tqdm(input_image):
        im_name = im.name

        nifti_name = im_name[:-5] + (
            "_0000.nii.gz" if append_index_suffix else ".nii.gz"
        )
        if (output_path / nifti_name).exists():
            continue
        else:
            convert_types(str(im), str(output_path / nifti_name))


def total_segmentator_predict_dir(case_dir, output_dir):
    """Run total segmentator on all nifti files in case_dir and save the results in output_dir."""
    output_dir.mkdir(exist_ok=True, parents=True)
    for im in tqdm(case_dir.iterdir()):
        im_name = im.name
        if not im_name.endswith(".nii.gz"):
            raise ValueError(f"File {im} is not a nifti file.")
        out_name = im_name[:-7] + "_seg.nii.gz"
        if (output_dir / out_name).exists():
            print("Skipping total segmentator creation as file exists already!")
            continue
        else:
            print(f"Running total segmentator on {im_name}")
            totalsegmentator(
                str(im),
                str(output_dir / out_name),
                nr_thr_resamp=3,
                nr_thr_saving=6,
                ml=True,
                fast=False,
                force_split=False,
            )


def anatomical_score_dir(case_dir, output_dir):
    """Run total segmentator on all nifti files in case_dir and save the results in output_dir."""
    output_dir.mkdir(exist_ok=True, parents=True)
    for im in tqdm(case_dir.iterdir()):
        im_name = im.name
        if not im_name.endswith(".nii.gz"):
            raise ValueError(f"File {im} is not a nifti file.")
        out_name = im_name[:-7] + "_seg.nii.gz"
        if (output_dir / out_name).exists():
            print("Skipping total segmentator creation as file exists already!")
            continue
        else:
            totalsegmentator(str(im), str(output_dir / out_name), ml=True)


def filter_non_same_size_cases(case: list[Path], label: list[Path]):
    mismatched_cases = []
    acceptable_cases_im = []
    acceptable_cases_lbl = []
    for c, l in zip(case, label):
        c_header = nrrd.read_header(str(c))
        l_header = nrrd.read_header(str(l))
        if np.any(c_header["sizes"] != l_header["sizes"]):
            mismatched_cases.append((c, l, c_header, l_header))
        else:
            acceptable_cases_im.append(c)
            acceptable_cases_lbl.append(l)
    # for mc in sorted(mismatched_cases, key=lambda x: x[0]):
    #     print(f"Case mismatch: {mc[0]} {mc[1]} - Shapes: {mc[2]['sizes']} {mc[3]['sizes']} - Spacings: \n{mc[2]['space directions']} \n{mc[3]['space directions']}")
    return acceptable_cases_im, acceptable_cases_lbl


def get_ids_from_dir(dir: Path):
    ids = []
    for case in dir.iterdir():
        name = case.name
        if not name.endswith(".nrrd"):
            continue
        ids.append(name.split("-")[-2])
    return list(set(ids))


def get_train_ids_and_im_path_from_dir(dir: Path) -> dict[int, Path]:
    """Intended vor the LNQ directory that contains both image and segmentation (and that other garbage)"""
    ids: dict[int, Path] = {}
    for case in dir.iterdir():
        name = case.name
        if "ct" in name:
            ids[int(name.split("-")[-2])] = case
    return ids


def get_ids_and_path_from_dir(dir: Path) -> dict[int, Path]:
    """Intended vor the LNQ directory that contains only the created segmentations"""
    ids: dict[int, Path] = {}
    for case in dir.iterdir():
        name = case.name
        ids[int(name.split("-")[-2])] = case
    return ids


def get_im_and_label_from_id(id: str, dir: Path, is_val: bool) -> tuple[Path, Path]:
    im = dir / f"lnq2023-{'val' if is_val else 'train'}-{id}-ct.nrrd"
    label = dir / f"lnq2023-{'val' if is_val else 'train'}-{id}-seg.nrrd"
    return im, label


def find_mutually_exclusive_classes_with_total_segmentator(
    total_segmentator_dir: Path, groundtruth_dir: Path, out_path: Path
):
    """Loads the both niftis, then compares the masks of all classes to the groundtruth mask. Calculates if classes overlap and if so, how much percent of the groundtruth is covered by the class."""
    if out_path.exists():
        out = load_json(str(out_path))
        mean_res = out["mean_res"]
        non_zero_mean = out["non_zero_mean"]
        all_results = out["all_results"]
    else:
        all_total_segmentator_files = list(sorted(os.listdir(total_segmentator_dir)))
        all_groundtruth_files = list(sorted(os.listdir(groundtruth_dir)))

        for ts, gt in zip(all_total_segmentator_files, all_groundtruth_files):
            assert ts.split("-")[2] == gt.split("-")[2], (
                "Total segmentator and groundtruth files do not match!"
                + f"N_Totalsegmentator: {len(all_total_segmentator_files)}, N_GT: {len(all_groundtruth_files)}"
                + f"Case ids: TS: {ts}, GT: {gt}"
            )

        all_results = {}
        for i in range(105):
            all_results[i] = []

        for ts, gt in tqdm(zip(all_total_segmentator_files, all_groundtruth_files)):
            ts_im = sitk.ReadImage(total_segmentator_dir / ts)
            gt_im = sitk.ReadImage(groundtruth_dir / gt)

            ts_data = sitk.GetArrayFromImage(ts_im)
            lymphnode_data = sitk.GetArrayFromImage(gt_im).astype(bool)
            n_lymphnode = np.count_nonzero(lymphnode_data)
            if n_lymphnode == 0:
                print("Empty groundtruth mask!")
                continue

            for i in range(105):
                ts_foreground_mask = ts_data == i
                all_results[i].append(
                    float(
                        np.sum(
                            np.logical_and(ts_foreground_mask, lymphnode_data),
                            dtype=float,
                        )
                        / float(n_lymphnode)
                    )
                )

        mean_res = {}
        for k, v in all_results.items():
            mean_res[k] = float(np.mean(v))

        non_zero_mean = {}
        for k, v in all_results.items():
            non_zero_mean[k] = float(
                np.mean(
                    np.mean([x for x in v if x != 0])
                    if len([x for x in v if x != 0]) != 0
                    else 0
                )
            )

        out_file = {
            "mean_res": mean_res,
            "non_zero_mean": non_zero_mean,
            "all_results": all_results,
        }
        save_json(out_file, str(out_path))

    return mean_res, non_zero_mean, all_results


def create_original_lnq_dataset(
    train_image_label_paths: list[Path],
    groundtruth_image_paths: list[Path],
    output_path: Path,
    dataset_name: str,
    dataset_json: dict,
):
    """Does not convert or anythin. Just moves files to the correct folder."""
    train_path = output_path / dataset_name / "imagesTr"
    label_path = output_path / dataset_name / "labelsTr"

    train_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)

    for train_im, train_label in zip(train_image_label_paths, groundtruth_image_paths):
        case_id = int(train_im.name.split("-")[-2])

        shutil.copy(train_im, train_path / (f"{case_id:04}_0000.nrrd"))
        shutil.copy(train_label, label_path / (f"{case_id:04}.nrrd"))

    save_json(dataset_json, output_path / dataset_name / "dataset.json")
    return


def convert_val_samples(val_dir: Path, val_out_dir: Path):
    all_files = [v for v in val_dir.iterdir() if v.name.endswith(".nrrd")]
    convert(all_files, val_out_dir, True)
    return


def create_nnunet_dataset(
    train_image_path: Path,
    groundtruth_image_path: Path,
    output_path: Path,
    dataset_name: str,
    dataset_json,
):
    """Moves the data from the chosen paths"""
    train_ids: dict[str, Path] = get_train_ids_and_im_path_from_dir(train_image_path)
    groundtruth_ids: dict[str, Path] = get_ids_and_path_from_dir(groundtruth_image_path)

    train_path = output_path / dataset_name / "imagesTr"
    label_path = output_path / dataset_name / "labelsTr"

    train_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)

    for ids in train_ids.keys():
        train_im = train_ids[ids]
        label_im = groundtruth_ids[ids]

        shutil.copy(train_im, train_path / f"{ids:04}_0000.nii.gz")
        shutil.copy(label_im, label_path / f"{ids:04}.nii.gz")
    save_json(dataset_json, output_path / dataset_name / "dataset.json")
    return


def simple_multidim_isin(arr1: np.ndarray, values: Sequence[int]):
    """Checks if all elements of arr1 are in arr2. arr1 can have more dimensions than arr2."""
    all_masks = []
    for val in values:
        all_masks.append(arr1 == val)
    mask = np.sum(np.stack(all_masks), axis=0) != 0
    return mask


def calculate_bpreg_treshold(path_to_bpreg, path_to_groundtruth)->tuple:
    bpreg_borders = {}
    # for each file in bpreg result find the matching groundtruth file
    for bpreg_file in path_to_bpreg.iterdir():
        #check if the file is a json file
        if bpreg_file.suffix != ".json":
            continue
        # find the matching groundtruth file
        groundtruth_file = path_to_groundtruth / bpreg_file.name.replace("_0000.json", "-seg.nii.gz").replace("_", "-")
        # load groundtuth file and find the indexes first and last of the sliecs that are not empty
        groundtruth_img = sitk.ReadImage(str(groundtruth_file))
        groundtruth = sitk.GetArrayFromImage(groundtruth_img)
        groundtruth = groundtruth.astype(int)
        # find the indexes of the first and last slice that are not empty
        zero_indices = np.nonzero(np.sum(groundtruth, axis=(1, 2)))
        first_slice = np.min(zero_indices)
        last_slice = np.max(zero_indices)
        # load the bpreg file and get the "cleaned dice score" value corresponding to the first and last slice
        pbreg = json.load(bpreg_file.open())
        try:
            min_bpreg = pbreg["cleaned slice scores"][first_slice]
            max_bpreg = pbreg["cleaned slice scores"][last_slice]  
        except IndexError:
            print(f"first slice: {first_slice}, last slice: {last_slice}")
            print(f"length of the bpreg slice scores: {len(pbreg['cleaned slice scores'])}")
            print(f"bpreg file: {bpreg_file}")
            print(f"groundtruth file: {groundtruth_file}")
            print(f"groundtruth shape: {groundtruth.shape}")
            raise IndexError
        # save the min and max values in a dictionary
        bpreg_borders[bpreg_file.name] = {"min": min_bpreg, "max": max_bpreg}
    
    # find the min and max values of the dictionary
    min_bpreg = np.min([bpreg_borders[key]["min"] for key in bpreg_borders.keys()])
    max_bpreg = np.max([bpreg_borders[key]["max"] for key in bpreg_borders.keys()])
    return min_bpreg, max_bpreg   

def calculate_background_borders_given_pbreg(path_to_bpreg, min_bpreg, max_bpreg):
    # for each file in bpreg result find the matching groundtruth file 
    background_borders_from_bpreg = {}
    for bpreg_file in path_to_bpreg.iterdir():
        #check if the file is a json file
        if bpreg_file.suffix != ".json":
            continue
        
        bpreg = json.load(bpreg_file.open())
        try:
            min_bpreg_index = int(np.where(bpreg["cleaned slice scores"] < min_bpreg)[0][-1])
            # add a margin of 2 slices to prevent outliers not being included if possible 
            if min_bpreg_index >= 2:
                min_bpreg_index -= 2
        except IndexError:
            min_bpreg_index = 0
        try:
            max_bpreg_index = int(np.where(bpreg["cleaned slice scores"] > max_bpreg)[0][0])
            # add a margin of 2 slices to prevent outliers not being included if possible 
            if max_bpreg_index <= len(bpreg["cleaned slice scores"]) - 3:
                max_bpreg_index += 2
        except IndexError:
            max_bpreg_index = len(bpreg["cleaned slice scores"]) - 1
        background_borders_from_bpreg[bpreg_file.name.replace("_0000.json", "-seg.nii.gz").replace("_", "-")] = {"min": min_bpreg_index, "max": max_bpreg_index}
        # dave to json file
    return background_borders_from_bpreg


def create_groundtruth_given_totalsegmentator(
    total_segmentator_dir: Path,
    total_segmentator_background_class_ids: Sequence[int],
    groundtruth_dir: Path,
    output_dir: Path,
    overwrite=False,
    make_outside_boundary_class=False,
    include_bpreg=False,
    background_borders_from_bpreg:dict[str, dict]=None,
    include_convex_hull=False,
    convex_hull_dir: Path=None,
    ignore_regions:dict[str, dict]=None,
) -> dict[str, int]:
    """Creates a groundtruth mask given the total segmentator segmentation and the ids of the background classes.
    Returns the dataset.json file."""
    all_total_segmentator_files = list(sorted(os.listdir(total_segmentator_dir)))
    all_groundtruth_files = list(sorted(os.listdir(groundtruth_dir)))
    output_dir.mkdir(exist_ok=True, parents=True)

    if make_outside_boundary_class:
        labels = {
            "background": 0,
            "lymphnode": 1,
            "lymphnode_outside_boundary": 2,
            "ignore": 3,
        }
    else:
        labels = {
            "background": 0,
            "lymphnode": 1,
            "ignore": 2,
        }

    for ts, gt in tqdm(zip(all_total_segmentator_files, all_groundtruth_files)):
        # Read total Segmentator segmentations
        output_path = output_dir / gt

        if output_path.exists():
            if not overwrite:
                continue

        ts_im = sitk.ReadImage(str(total_segmentator_dir / ts))
        ts_data = sitk.GetArrayFromImage(ts_im)
        ts_data = ts_data.astype(int)

        # Read groundtruth
        lnq_im = sitk.ReadImage(str(groundtruth_dir / gt))
        lnq_data = sitk.GetArrayFromImage(lnq_im)
        lnq_data = lnq_data.astype(int)

        # Create final groundtruth (all total_segmentator_background_class_ids + 1 voxel from boundary are background, all lymph nodes are foreground, rest is ignore)
        final_groundtruth = np.full_like(
            lnq_data, fill_value=labels["ignore"]
        )  # 0 will be background, 1 is ignore label and 2 is foreground (lymph node)

        total_segmentator_mask = simple_multidim_isin(
            ts_data, total_segmentator_background_class_ids
        )  # This is slow but running it once is enough, so who cares


        # if bodypartregression is to be be included add the voxels outside the bodypartregression boundaries to the background
        if include_bpreg:
            final_groundtruth[:background_borders_from_bpreg[gt]["min"]] = labels["background"]
            final_groundtruth[background_borders_from_bpreg[gt]["max"]:] = labels["background"]
        
        # set all areas outside the convex hull of the lung to background
        if include_convex_hull:
            file_name = gt.replace("seg.nii.gz", "ct_seg.nii.gz")
            convex_hull = sitk.GetArrayFromImage(sitk.ReadImage(str(convex_hull_dir / file_name)))
            final_groundtruth = np.where(
                convex_hull == 0,
                labels["background"],
                final_groundtruth,
            )

        # Set all total segmentator predicted classes (that we want to set to background) to background
        final_groundtruth = np.where(
            total_segmentator_mask, labels["background"], final_groundtruth
        )
        # To be safe of overlaps we set the lymph node (foreground) after to foreground
        final_groundtruth = np.where(
            lnq_data != 0, labels["lymphnode"], final_groundtruth
        )
        # for error correction we ignore some slices around the start and end of the lung region
        if include_convex_hull:
            # set regions from ignore_regions to ignore label 
            ignore_region = ignore_regions[gt]
            final_groundtruth[ignore_region["min_from"]:ignore_region["min_to"]] = labels["ignore"]
            final_groundtruth[ignore_region["max_from"]:ignore_region["max_to"]] = labels["ignore"]

        final_groundtruth = final_groundtruth.reshape(lnq_data.shape)

        lnq_binary = np.where(lnq_data != 0, 1, 0)  # Make binary
        dilated_lnq_binary = morphology.binary_dilation(
            lnq_binary, iterations=2
        )  # Dilate with Square connectivity equal to one
        lnq_boundary = (
            dilated_lnq_binary - lnq_binary
        )  # This is the boundary of the object

        if make_outside_boundary_class:
            final_groundtruth = np.where(
                lnq_boundary != 0,
                labels["lymphnode_outside_boundary"],
                final_groundtruth,
            )
        else:
            final_groundtruth = np.where(
                lnq_boundary != 0, labels["background"], final_groundtruth
            )

        output_im = sitk.GetImageFromArray(final_groundtruth.astype(np.uint32))
        output_im.CopyInformation(lnq_im)
        sitk.WriteImage(output_im, str(output_path))

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": len(all_total_segmentator_files),
        "file_ending": ".nii.gz",
    }

    return dataset_json


def calculate_ignore_region_around_lung(bpreg_dir: Path):
    lung_start = 44.143
    lung_end = 75.389
    distance = 5
    # for each file in bpreg result find the matching groundtruth file 
    ignore_region_from_bpreg = {}
    for bpreg_file in bpreg_dir.iterdir():
        #check if the file is a json file
        if bpreg_file.suffix != ".json":
            continue
        
        # calculate the index until where cleaned slice scores are smaller than the lung start
        bpreg = json.load(bpreg_file.open())
        try:
            min_bpreg_index = np.where(np.array(bpreg["cleaned slice scores"]) < lung_start)[0][-1]
        except IndexError:
            min_bpreg_index = 0
        try:
            max_bpreg_index = np.where(np.array(bpreg["cleaned slice scores"]) > lung_end)[0][0]
        except IndexError:
            max_bpreg_index = len(bpreg["cleaned slice scores"]) - 1
        ignore_region_from_bpreg[bpreg_file.name.replace("_0000.json", "-seg.nii.gz").replace("_", "-")] = {"min_from": min_bpreg_index - distance,"min_to":min_bpreg_index + distance, "max_from": max_bpreg_index - distance, "max_to": max_bpreg_index + distance}   
        
    return ignore_region_from_bpreg

def main():
    """
    DISCLAIMER:

    This is a necessary preprocessing step as the annotation scheme of the LNQ challenge is so weird.
    So in order to get some (guaranteed) background class voxels inferences is done of total segmentator.
    Predicted classes that should not overlap with the lymph nodes are used to create actual background classes.

    Additionally to that the Lymphnodes are region-grown (by a little bit) and that area is set as negative clas as well.

    Everything else will be set to ignore label, as it might as well be a lymph node, but we do not know.

    Finally postprocessing will have to deal with the removal of too small lymph nodes as obviously the challenge organizers use the weird 2D radiomics garbage to determine the biggest diameters...

    !!!!!!!!!!
    WARNING: This has to be run in a separate environment to the normal nnUNet prepocessing as TotalSegmentator is incompatible (especially to nnUNet V2!)
    !!!!!!!!!!

    So run this in a totalsegmentator env first, then run the Dataset911_LNQ.py conversion which will need the temporary files created here.
    """

    path_to_data = Path("/home/j683r/local-work-temporary/LNQ")
    
    meta_info_path = path_to_data / "meta_info.json"
    temp_in_path = path_to_data / "total_segmentator_LNQ" / "in"
    temp_lbl_path = path_to_data / "total_segmentator_LNQ" / "lbl"
    temp_out_path = path_to_data / "total_segmentator_LNQ" / "seg"
    val_path = path_to_data / "val"
    val_nifti_path = path_to_data / "val_nifti"

    convert_val_samples(val_path, val_nifti_path)

    out_dir_aorta_bpreg = path_to_data / "background_no_aorta_bpreg"

    nnunet_raw_data_path = Path(os.environ["nnUNet_raw"])
    print(f"nnUNet raw data path: {nnunet_raw_data_path}  ")
    train_dir = (
        path_to_data / "patched_train"
    )  # Contains the new segmentation labels that are (hopefully) the same shape as original labels.

    train_ids = get_ids_from_dir(train_dir)

    train_im_labels = [
        get_im_and_label_from_id(id, train_dir, False) for id in train_ids
    ]

    # Assure the train cases and labels have same shape.
    only_train_images = [ti[0] for ti in train_im_labels]
    only_train_labels = [tl[1] for tl in train_im_labels]
    remaining_cases, remaining_labels = filter_non_same_size_cases(
        only_train_images, only_train_labels
    )

    convert(remaining_cases, temp_in_path)
    convert(remaining_labels, temp_lbl_path)
    total_segmentator_predict_dir(temp_in_path, temp_out_path)
    

    (
        mean_res,
        non_zero_mean,
        all_results,
    ) = find_mutually_exclusive_classes_with_total_segmentator(
        temp_out_path, temp_lbl_path, meta_info_path
    )

    background_classes_no_aorta = {
        int(k) for k, v in non_zero_mean.items() if not (int(k) in [0, 7])
    }  # 0 is background, so we do not want that and do not want classes that overlap with lymphnodes.
    
    if (path_to_data / "bodypartregression").exists():
        #calculate the bodypartregression tresholds
        min_bpreg, max_bpreg = calculate_bpreg_treshold(path_to_data / "bodypartregression", temp_lbl_path)
        #calculate the background borders from the bodypartregression results
        background_borders_from_bpreg = calculate_background_borders_given_pbreg(path_to_data / "bodypartregression", min_bpreg, max_bpreg)
    else:
        background_borders_from_bpreg = json.load(open("background_borders_from_bpreg.json", "r"))


    print("No Aorta bpreg: Create groundtruth given total segmentator and bodypartregression")
    no_aorta_and_bpreg_dataset_json = create_groundtruth_given_totalsegmentator(
        temp_out_path,
        background_classes_no_aorta,
        temp_lbl_path,
        out_dir_aorta_bpreg, 
        include_bpreg=True, 
        background_borders_from_bpreg=background_borders_from_bpreg
    )



    # Now we create the nnUNet compatible datasets


    print("Create Dataset915_aorta_not_background_with_bpreg.")
    create_nnunet_dataset(
        train_image_path=temp_in_path,
        groundtruth_image_path=out_dir_aorta_bpreg,
        output_path=nnunet_raw_data_path,
        dataset_name="Dataset915_aorta_not_background_with_bpreg",
        dataset_json=no_aorta_and_bpreg_dataset_json,
    )


    sys.exit(0)


if __name__ == "__main__":
    main()
