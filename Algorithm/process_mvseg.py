from copy import deepcopy
import inspect
import numpy as np
import os
from pathlib import Path
import torch
from typing import Union, List

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def preprocess_from_file(input_files: List[List[str]],
                        output_filenames_truncated: Union[None, str],
                        plans_manager: PlansManager,
                        dataset_json: dict,
                        configuration_manager: ConfigurationManager,
                        device: torch.device,
                        verbose: bool = False):
    preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
    data, _, data_properites = preprocessor.run_case(input_files, None, plans_manager, configuration_manager, dataset_json)

    data = torch.from_numpy(data).contiguous().float().to(device=device)

    item = {'data': data, 'data_properites': data_properites,
            'ofile': output_filenames_truncated if output_filenames_truncated is not None else None}

    return item

class Predictor(nnUNetPredictor):
    def __init__(self,
                    tile_step_size: float = 0.5,
                    use_gaussian: bool = True,
                    use_mirroring: bool = True,
                    perform_everything_on_gpu: bool = True,
                    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    verbose: bool = False,
                    verbose_preprocessing: bool = False,
                    allow_tqdm: bool = True):
        super().__init__(tile_step_size, use_gaussian, use_mirroring,
                    perform_everything_on_gpu, device, verbose,
                    verbose_preprocessing, allow_tqdm)

    def predict_from_lists_of_filenames(self, list_of_lists, output_filename_truncated):
        for i in range(len(list_of_lists)):
            preprocessed = preprocess_from_file(list_of_lists[i], output_filename_truncated[i], 
                                                    self.plans_manager, self.dataset_json,
                                                    self.configuration_manager, self.device,
                                                    self.verbose_preprocessing)
    
            data = preprocessed['data']
            if isinstance(data, str):
                delfile = data
                data = torch.from_numpy(np.load(data))
                os.remove(delfile)

            ofile = preprocessed['ofile']
            if ofile is not None:
                print(f'\nPredicting {os.path.basename(ofile)}:')
            else:
                print(f'\nPredicting image of shape {data.shape}:')

            print(f'perform_everything_on_gpu: {self.perform_everything_on_gpu}')

            properties = preprocessed['data_properites']

            # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
            # npy files

            prediction = self.predict_logits_from_preprocessed_data(data).cpu()

            export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                                        self.dataset_json, ofile)
            print(f'done with {os.path.basename(ofile)}')

    def predict_from_files(self,
                            list_of_lists_or_source_folder: Union[str, List[List[str]]],
                            output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                            save_probabilities: bool = False,
                            overwrite: bool = True,
                            folder_with_segs_from_prev_stage: str = None,
                            num_parts: int = 1,
                            part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, _ = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        return self.predict_from_lists_of_filenames(list_of_lists_or_source_folder, output_filename_truncated)


def predict(input_path="/input", output_path="/output"):
    input_path = Path(input_path)
    output_path = Path(output_path)
    predictor = Predictor()
    predictor.initialize_from_trained_model_folder("/nnUNet_results/Dataset918_aorta_not_background_with_bpreg_and_additional_data_fixed/nnUNetTrainerDA5__resenc_planner__3d_fullres/",
                                                    use_folds=(4,), 
                                                    checkpoint_name='checkpoint_final.pth')
    input_files = sorted([[f] for f in input_path.iterdir() if f.match("*.nii.gz")])
    output_files = [output_path/f"{f[0].name[:-7]}_seg.nii.gz" for f in input_files]

    input_files = [[str(f[0])] for f in input_files]
    output_files = [str(f) for f in output_files]

    predictor.predict_from_files(input_files, output_files, save_probabilities=False,
                                overwrite=False, folder_with_segs_from_prev_stage=None,
                                num_parts=1, part_id=0)


if __name__ == "__main__":
    print("Start prediction")

    predict()

    print("Done!")
