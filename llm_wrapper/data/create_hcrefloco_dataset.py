"""
instruction for the HC-RefLoCo dataset
https://github.com/ZhaoJingjing713/HC-RefLoCo

download the dataset in ./llm_wrapper/data:
sudo apt install git-lfs
git clone https://huggingface.co/datasets/Jinjing713/HC-RefLoCo

Install the dataset:
pip install 'git+https://github.com/ZhaoJingjing713/HC-RefLoCo.git'

You should have images in ./llm_wrapper/data/HC_RefLoCo/HC_RefLoCo_images folder.

The readme file details the usage of this script.
"""

import json
import os

import sys
sys.path.append("HC-RefLoCo")

from llm_wrapper.data.HC-RefLoCo.hc_refloco import HCRefLoCoDataset
from tqdm import tqdm

from llm_wrapper.config_llm_wrapper import get_args


def main(args):

    datadir = os.path.join(args.main_folder, "HC_RefLoCo")
    os.makedirs(os.path.join(datadir, "dataset_json"), exist_ok=True)

    key_to_key = {
        "GT_bbox": "bbox",
        "file_name": "file_name",
        "height": "height",
        "width": "width",
        "sent_id": "id",
        "sent_text": "caption",
    }

    db_json = {"val": [], "test": []}

    dts = HCRefLoCoDataset(
        dataset_path=os.path.join(args.main_folder, "HC-RefLoCo"),
        split="all",
        load_img=False,
        images_file="images.tar.gz",
    )

    for i in tqdm(range(len(dts))):
        _, item = dts[i]
        sample = {k: item[v] for k, v in key_to_key.items()}
        db_json[item["split"]].append(sample)

    print("saving json files")
    print("Val samples:", len(db_json["val"]))
    print("Test samples:", len(db_json["test"]))

    for split in db_json.keys():
        with open(
            f"{datadir}/dataset_json/HC_RefLoCo_samples_for_my_dataset_{split}.json",
            "w",
        ) as f:
            json.dump(db_json[split], f, indent=4)
    return None


if __name__ == "__main__":

    args = get_args()
    main(args)
