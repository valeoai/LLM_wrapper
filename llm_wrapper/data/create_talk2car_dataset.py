"""
git clone https://github.com/talk2car/Talk2Car.git

Then checkout what requirements you do not already have installed and install them.

cat Talk2Car/requirements.txt

Some that you may not have:
- spacy
- nuscenes-devkit

Run:
python -m spacy download en_core_web_sm

This file contains an example of a dataloader for the Talk2Car dataset
(You need nuscenes dataset images downloaded in ./llm_wrapper/data/nuscenes)

The readme file details the usage of this script.
"""

import json
import os
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
sys.path.append("Talk2Car")

from llm_wrapper.config_llm_wrapper import get_args
from llm_wrapper.data.Talk2Car.talk2car import get_talk2car_class
from llm_wrapper.data.Talk2Car.vocabulary import Vocabulary


# An example dataset implementation
class ExampleDataset(Dataset):
    def __init__(self, root, vocabulary, command_path=None, version="train", slim_t2c=True):
        # Initialize
        self.version = version
        self.vocabulary = vocabulary
        self.dataset = get_talk2car_class(root, split=self.version, slim=slim_t2c, command_path=command_path)

        # Fixed params
        self.num_classes = 23
        self.class_names = [
            "animal",
            "human.pedestrian.adult",
            "human.pedestrian.child",
            "human.pedestrian.construction_worker",
            "human.pedestrian.personal_mobility",
            "human.pedestrian.police_officer",
            "human.pedestrian.stroller",
            "human.pedestrian.wheelchair",
            "movable_object.barrier",
            "movable_object.debris",
            "movable_object.pushable_pullable",
            "movable_object.trafficcone",
            "static_object.bicycle_rack",
            "vehicle.bicycle",
            "vehicle.bus.bendy",
            "vehicle.bus.rigid",
            "vehicle.car",
            "vehicle.construction",
            "vehicle.emergency.ambulance",
            "vehicle.emergency.police",
            "vehicle.motorcycle",
            "vehicle.trailer",
            "vehicle.truck",
        ]
        self.class_name_to_id = dict(map(reversed, enumerate(self.class_names)))
        self.id_to_class_name = dict(enumerate(self.class_names))

    def __len__(self):
        return len(self.dataset.commands)

    def __getitem__(self, index):
        # Command
        command = self.dataset.commands[index]
        descr = torch.Tensor(self.vocabulary.sent2ix_andpad(command.command, add_eos_token=True)).long()
        text = self.vocabulary.ix2sent_drop_pad(descr.numpy().tolist())
        text = " ".join(text[:-1]).replace(" ,", ",").replace(" .", ".")

        # Get paths
        sd_rec = self.dataset.get("sample_data", command.frame_token)
        img_path, _, _ = self.dataset.get_sample_data(sd_rec["token"])

        img = Image.open(img_path)

        # Bbox [x0,y0,w,h]
        bbox = list(map(int, command.get_2d_bbox()))

        return img_path, img, text, bbox


def add_img_ids_sent_ids(data, dico_img_ids, dico_sent_ids):

    for it, elem in enumerate(data):
        sent_text = elem["sent_text"]
        filepath = elem["file_name"]

        elem["sent_id"] = int(dico_sent_ids[sent_text])
        elem["image_id"] = int(dico_img_ids[filepath])

    return data


def main(args):

    # Dataroot
    root = os.path.join(args.main_folder, "nuscenes")
    save_path = os.path.join(args.main_folder, "talk2car", "dataset_json")
    os.makedirs(save_path, exist_ok=True)

    # Vocabulary
    vocabulary = Vocabulary("./Talk2Car/t2c_voc.txt")

    db_json = defaultdict(list)

    for version in ["test", "train", "val"]:
        # Retrieve dataset
        dataset = ExampleDataset(root, vocabulary, command_path="./", version=version, slim_t2c=False)
        print("Number of %s samples is %d" % (version, len(dataset)))

        for j in tqdm(range(len(dataset)), desc=f"Creating {version} dataset"):
            try:
                img_path, img, descr, bbox = dataset[j]
                width, height = img.size
                sample = {
                    "file_name": img_path.replace(args.main_folder, "").strip(
                        "/"
                    ),  # ensure file_names start with "nuscenes/..."
                    "sent_text": descr,
                    "GT_bbox": bbox,
                    "height": height,
                    "width": width,
                    "id": str(j),
                }
                db_json[version].append(sample)
            except Exception as e:
                print(f"Error: {e}")

    # Create two additional dictionaries to identify uniquely image ids and sentence ids

    # keys are img full filepaths and values img_ids
    dico_img_ids = {}
    # keys are sent texts and values sent_ids
    dico_sent_ids = {}

    sent_id_to_use, img_id_to_use = 0, 0
    for data in [db_json["val"], db_json["test"], db_json["train"]]:
        for it, elem in enumerate(data):
            sent_text = elem["sent_text"]
            filepath = elem["file_name"]
            if sent_text not in dico_sent_ids.keys():
                dico_sent_ids[sent_text] = sent_id_to_use
                sent_id_to_use += 1
            if filepath not in dico_img_ids.keys():
                dico_img_ids[filepath] = img_id_to_use
                img_id_to_use += 1

    db_json["val"] = add_img_ids_sent_ids(data=db_json["val"], dico_img_ids=dico_img_ids, dico_sent_ids=dico_sent_ids)
    db_json["test"] = add_img_ids_sent_ids(data=db_json["test"], dico_img_ids=dico_img_ids, dico_sent_ids=dico_sent_ids)
    db_json["train"] = add_img_ids_sent_ids(data=db_json["train"], dico_img_ids=dico_img_ids, dico_sent_ids=dico_sent_ids)

    for version in ["test", "train", "val"]:
        with open(
            os.path.join(save_path, f"talk2car_samples_for_my_dataset_{version}.json"),
            "w",
        ) as f:
            json.dump(db_json[version], f, indent=4)

    return None


if __name__ == "__main__":

    args = get_args()
    main(args)
