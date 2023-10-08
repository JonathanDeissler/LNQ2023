from pathlib import Path
import shutil
import SimpleITK
import os
import sys
import process_mvseg
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)


class Lnq2023(SegmentationAlgorithm):
    def __init__(self):
        output_path = Path('/output/images/mediastinal-lymph-node-segmentation/')
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        super().__init__(
            input_path=Path('/input/images/mediastinal-ct/'),
            output_path=output_path,
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
    def empty_folder(self,folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        # make temporary directory for input and output
        input_path = Path('/temporary/inputs/')
        output_path = Path('/temporary/outputs/')
        output_path_pp = Path('/temporary/outputs_pp/')
        shutil.copy('/nnUNet_results/Dataset918_aorta_not_background_with_bpreg_and_additional_data_fixed/nnUNetTrainerDA5__resenc_planner__3d_fullres/dataset.json',output_path)
        # save input image
        SimpleITK.WriteImage(input_image, str(input_path / 'lnq_01_0000.nii.gz'))
        # run algorithm
        process_mvseg.predict(str(input_path), str(output_path))

        output_image = SimpleITK.ReadImage(str(output_path / 'lnq_01_0000_seg.nii.gz'))
        output_image.CopyInformation(input_image)
        #remove files from temporary directory
        
        # output_image = SimpleITK.ReadImage(str(input_path / 'lnq_01_0000.nii.gz'))
        self.empty_folder(input_path)
        self.empty_folder(output_path)
        self.empty_folder(output_path_pp)
        return output_image

if __name__ == "__main__":
    Lnq2023().process()

