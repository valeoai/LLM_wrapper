"""
cd ./llm_wrapper/data:
git clone https://github.com/lichengunc/refer

Install it following instructions given in (github.com/lichengunc/refer)
Then checkout what requirements you do not already have installed and install them.
(You need COCO images in ./llm_wrapper/data/COCO/images/train2014 folder.)

The readme file details the usage of this script.
"""

import json
import os
import sys

from llm_wrapper.config_llm_wrapper import get_args

sys.path.append("refer")
from llm_wrapper.data.refer.refer import REFER


def extract_dataset(dataset, splitBy, split, return_list_of_image_ids=False):

    data_root = "refer/data"
    refer = REFER(data_root, dataset=dataset, splitBy=splitBy)  # only possible split for refcoco+ (no overlap btw images)

    list_of_samples = []
    list_of_image_ids = []
    ref_ids = refer.getRefIds(split=split)
    # print("Total nb of refs:", len(ref_ids))

    for idx in range(len(ref_ids)):

        ref_id = ref_ids[idx]
        ref = refer.Refs[ref_id]
        image_id = ref["image_id"]
        list_of_image_ids.append(image_id)
        # get image info
        image_info = refer.loadImgs(image_ids=[image_id])[0]
        GT_bbox = refer.getRefBox(ref_id=ref_id)

        # get sentences info
        raw_sents = [elem["raw"] for elem in ref["sentences"]]
        # print("raw_sents:", raw_sents)
        sent_ids = [elem["sent_id"] for elem in ref["sentences"]]

        for sent_text, sent_id in zip(raw_sents, sent_ids):
            info_dict = {}
            info_dict["image_id"] = image_id
            info_dict["height"] = image_info["height"]
            info_dict["width"] = image_info["width"]
            info_dict["file_name"] = image_info["file_name"]
            info_dict["GT_bbox"] = GT_bbox
            info_dict["sent_text"] = sent_text.lower().strip()
            info_dict["sent_id"] = sent_id

            list_of_samples.append(info_dict)

    print("len list_of_samples:", len(list_of_samples))

    if return_list_of_image_ids:
        list_of_image_ids = list(set(list_of_image_ids))
        return list_of_samples, list_of_image_ids

    else:
        return list_of_samples


def main(args):

    for dataset_name in ["refcoco", "refcoco+", "refcocog"]:

        if dataset_name == "refcoco":
            splitBy = "unc"
            full_dataset_name = f"RefCOCO_{splitBy}"
        elif dataset_name == "refcoco+":
            splitBy = "unc"
            full_dataset_name = f"RefCOCO+_{splitBy}"
        elif dataset_name == "refcocog":
            splitBy = "umd"
            full_dataset_name = f"RefCOCOg_{splitBy}"

        os.makedirs(
            os.path.join(args.main_folder, full_dataset_name, "dataset_json"),
            exist_ok=True,
        )

        for split in ["val", "test", "train"]:

            print(f"\nExtracting dataset {dataset_name} / split {split}")
            list_of_samples = extract_dataset(
                dataset=dataset_name,
                splitBy=splitBy,
                split=split,
                return_list_of_image_ids=False,
            )
            dataset_dict_path = os.path.join(
                args.main_folder,
                full_dataset_name,
                "dataset_json",
                f"{full_dataset_name}_samples_for_my_dataset_{split}.json",
            )
            with open(dataset_dict_path, "w") as outfile:
                json.dump(list_of_samples, outfile, indent=4)

    return None


if __name__ == "__main__":

    args = get_args()
    main(args)
