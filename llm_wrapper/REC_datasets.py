import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class my_REC_dataset(Dataset):
    def __init__(
        self,
        main_folder,
        split,
        dataset_name,
        detector_name=None,
        return_text_data_only=False,
        verbose=False,
    ):

        print(f"\nInit {dataset_name} for split = {split}")

        self.main_folder = main_folder
        self.dataset_name = dataset_name
        self.return_text_data_only = return_text_data_only

        # Define image output type

        if detector_name is None and detector_name != "KOSMOS":
            self.image_output_type = None
        elif detector_name == "KOSMOS":
            self.image_output_type = "img_path"
        elif detector_name in [
            "GDINO_SwinT",
            "GDINO_SwinB",
            "GDINO_SwinT_original",
            "GDINO_SwinB_original",
        ]:
            # import torchvision.transforms.functional as F
            # from utils_for_GD import (
            #     Compose,
            #     FixResize,
            #     Normalize,
            #     ToTensor,
            #     box_xyxy_to_cxcywh,
            #     modif_load_image,
            # )

            self.image_output_type = "GD_GDrec_img_loading"
        else:
            self.image_output_type = "PIL_img"

        # Define images path

        if self.dataset_name in ["RefCOCOg_umd", "RefCOCO_unc", "RefCOCO+_unc"]:
            self.IMAGE_DIR = os.path.join(main_folder, "COCO", "images", "train2014")

        elif self.dataset_name == "talk2car":
            self.IMAGE_DIR = main_folder

        elif self.dataset_name == "HC_RefLoCo":
            self.IMAGE_DIR = os.path.join(main_folder, "HC_RefLoCo", "HC_RefLoCo_images")

        # Get path to our custom REC dataset data formatting : (image_path, text query) pairs

        dataset_dict_path = os.path.join(
            main_folder,
            dataset_name,
            "dataset_json",
            f"{dataset_name}_samples_for_my_dataset_{split}.json",
        )
        with open(dataset_dict_path) as f1:
            self.dataset = json.load(f1)

        if verbose:
            print("REC dataset length", len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        # Define image path and image id

        info_dict = self.dataset[
            idx
        ]  # dict with keys: 'image_id', 'file_name', 'width', 'height', 'GT_bbox', 'sent_id', 'sent_text'
        img_path = os.path.join(self.IMAGE_DIR, info_dict["file_name"])
        image_id = info_dict["file_name"] if self.dataset_name in ["HC_RefLoCo"] else info_dict["image_id"]

        # Return proper output

        if self.return_text_data_only:
            return int(info_dict["sent_id"]), info_dict["sent_text"]

        if self.image_output_type == "img_path":
            return (
                image_id,
                img_path,
                int(info_dict["sent_id"]),
                info_dict["sent_text"],
                np.array(info_dict["GT_bbox"]),
            )

        elif self.image_output_type == "PIL_img":
            raw_image = Image.open(img_path)
            whwh = None
            return (
                image_id,
                raw_image,
                whwh,
                int(info_dict["sent_id"]),
                info_dict["sent_text"],
                np.array(info_dict["GT_bbox"]),
            )

        elif self.image_output_type == "GD_GDrec_img_loading":
            from utils_for_GD import modif_load_image

            _, image = modif_load_image(img_path)
            whwh = np.array(
                [
                    info_dict["width"],
                    info_dict["height"],
                    info_dict["width"],
                    info_dict["height"],
                ]
            )
            return (
                image_id,
                image,
                whwh,
                int(info_dict["sent_id"]),
                info_dict["sent_text"],
                np.array(info_dict["GT_bbox"]),
            )  # GT_bbox format is [x, y, w, h]

        elif self.image_output_type is None:
            whwh = np.array(
                [
                    info_dict["width"],
                    info_dict["height"],
                    info_dict["width"],
                    info_dict["height"],
                ]
            )
            return (
                image_id,
                whwh,
                int(info_dict["sent_id"]),
                info_dict["sent_text"],
                np.array(info_dict["GT_bbox"]),
            )  # GT_bbox format is [x, y, w, h]


class dataset_for_step_3_various_detectors(Dataset):
    def __init__(self, filepath, filepath_bis=None):

        # all_info_step_2 is a list of len the nb of data preprocessed (by VLM)

        self.all_info_step_2 = torch.load(filepath)

        self.filepath_bis = filepath_bis
        if filepath_bis is not None:
            self.all_info_step_2_bis = torch.load(filepath_bis)

    def __len__(self):
        return len(self.all_info_step_2)

    def __getitem__(self, idx):
        if self.filepath_bis is None:
            return self.all_info_step_2[idx]
        else:
            return self.all_info_step_2[idx], self.all_info_step_2_bis[idx]
