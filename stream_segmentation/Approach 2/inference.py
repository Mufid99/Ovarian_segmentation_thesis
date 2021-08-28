
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.model_restore import load_model_and_checkpoint_files
import nnunet.inference.predict as prd


class Predictor:
    def __init__(self):
        self.folds = None
        self.all_in_gpu = False
        self.mixed_precision = True
        self.model = './model_files_required/nnUNetTrainerV2__nnUNetPlansv2.1'
        self.expected_num_modalities = load_pickle(join(self.model, "plans.pkl"))['num_modalities']
        self.lowres_segmentations = None
        self.checkpoint_name = "model_final_checkpoint"
        self.trainer, self.params = load_model_and_checkpoint_files(self.model, self.folds, mixed_precision=self.mixed_precision,
                                                        checkpoint_name=self.checkpoint_name)

    # predict segmentation of image providing the method.
    def predict(self, input, ensemble=False, do_tta = False):
        return prd.predict_case(input, self.folds, self.trainer, self.params, ensemble=ensemble, do_tta=do_tta)
    

