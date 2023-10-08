# Repo for the LNQ challenge

Challenge info:

The challenge deals with lymph node quantification (segmentation) but has a catch: Not all Lymph nodes are annotated within a patient.
Out of e.g. 5 Lymphnodes (LN) only a single one is annotated.

to recreate the submitted model:
-  Download data
-  Preprocess data
    -calculate bodypart regression scores using the run_inference_on_bodypartregression.py (alternatively if only using the challenge data you can skip this stem and use the precomputed bodypartregression values provided)
    -Dataset preprocessing can be found in Dataset918_LNQ.py
-  convert the additional dataset using the convert_additional_dataset.py script 
-  copy the converted file into the raw nnU-Net dataset (make sure to modify the dataset.json as well to account for the added cases)
-  Run the plan function of nnU-Net using the resenc planner (nUNetv2_plan_experiment -d <DatasetID> -c 3d_fullres -pl ResEncUNetPlanner -overwrite_plans_name resenc_planner)
-  Modify the resulting plans file to incorporate:
    "batch_size": 4,
            "patch_size": [
                128,
                192,
                192
            ]
-  Run nnU-Net preprocessing (nnUNetv2_preprocess -d <DatasetID> -plans_name resenc_planner -c 3d_fullres)
-  Train nnU-Net using the trainer: nnUNetTrainerDA5
-  Inference can be run via the nnUNetv2_predict function

-  For containerisation see the code provided in Algorithm

Installation help for the used tools can be found here:
    https://github.com/MIC-DKFZ/nnUNet
    https://github.com/MIC-DKFZ/BodyPartRegression
    https://github.com/wasserth/TotalSegmentator
    
