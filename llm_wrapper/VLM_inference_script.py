import json
import os

import torch
from torch.utils.data import DataLoader

from llm_wrapper.config_llm_wrapper import get_args
from llm_wrapper.inference_florence_2 import postpro_step_2_florence2, prepro_step_2_flo2
from llm_wrapper.naming_utils import get_VLM_preprocessed_files_paths
from llm_wrapper.REC_datasets import my_REC_dataset

# Not released yet
step_1_with_spacy = None
prepro_step_2_GD_GDrec = None
postpro_step_2_GDrec = None
postpro_step_2_GD_ori = None
postpro_step_2_gdino_1_5 = None
postpro_step_2_KOSMOS = None


def main(args):

    # Get filepath name to use to save or load preprocessed data using VLM(s)
    prepro_data_filepaths = get_VLM_preprocessed_files_paths(args=args)

    # Case where VLM preprocessing is needed
    if not os.path.exists(prepro_data_filepaths["step_2"]) and not os.path.exists(prepro_data_filepaths["full_step_2"]):

        # Text preprocessing (load or infer) if use GD/GDrec
        if args.detector_name in [
            "GDINO_SwinT",
            "GDINO_SwinB",
            "GDINO_SwinT_original",
            "GDINO_SwinB_original",
        ]:

            if os.path.exists(prepro_data_filepaths["step_1"]):
                print("\nText preprocessing already done")
                with open(prepro_data_filepaths["step_1"]) as f:
                    dico_step_1 = json.load(f)
            elif os.path.exists(prepro_data_filepaths["full_step_1"]):
                print("\nText preprocessing already done")
                with open(prepro_data_filepaths["full_step_1"]) as f:
                    dico_step_1 = json.load(f)
            else:
                dico_step_1 = step_1_with_spacy(args=args)
                with open(prepro_data_filepaths["step_1"], "w") as outfile:
                    json.dump(dico_step_1, outfile, indent=4)
                print(
                    "Done dumping results from text preprocessing (with spacy) into folder",
                    prepro_data_filepaths["step_1"],
                )

        # Text-image pairs preprocessing (basic VLM inference)
        if not args.not_vlm_prepro:

            rec_dataset = my_REC_dataset(
                main_folder=args.main_folder,
                split=args.split,
                dataset_name=args.dataset,
                detector_name=args.detector_name,
            )

            if args.detector_name != "GDINO_1.5":
                rec_dataset = torch.utils.data.Subset(rec_dataset, range(args.min_samples, args.max_samples))

            if args.detector_name in [
                "GDINO_SwinT",
                "GDINO_SwinB",
                "GDINO_SwinT_original",
                "GDINO_SwinB_original",
            ]:
                loader = DataLoader(
                    rec_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                prepro_step_2_GD_GDrec(args=args, loader=loader)

            elif args.detector_name in [
                "Florence-2-large-v2",
                "Florence-2-large-ft-v2",
            ]:
                prepro_step_2_flo2(args=args, data=rec_dataset)

            elif "GDINO_1.5" in args.detector_name:
                print("Preprocessing not implemented for GDINO 1.5 as it was done via API calls")

            else:
                raise Exception(f"Preprocessing not implemented for {args.detector_name}")

        # Text-image pairs postprocessing (setting outputs to desired format for LLM-wrapper)
        if not args.not_vlm_postpro:
            rec_dataset = my_REC_dataset(
                main_folder=args.main_folder,
                split=args.split,
                dataset_name=args.dataset,
            )

            if args.detector_name != "GDINO_1.5":
                rec_dataset = torch.utils.data.Subset(rec_dataset, range(args.min_samples, args.max_samples))
            print("len dataset:", len(rec_dataset))
            loader = DataLoader(
                rec_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            if args.detector_name in ["GDINO_SwinT", "GDINO_SwinB"]:
                _ = postpro_step_2_GDrec(
                    args=args,
                    loader=loader,
                    dico=dico_step_1,
                    save_postpro_filename=prepro_data_filepaths["step_2"],
                )

            elif args.detector_name in ["GDINO_SwinT_original", "GDINO_SwinB_original"]:
                _ = postpro_step_2_GD_ori(
                    args=args,
                    loader=loader,
                    dico=dico_step_1,
                    save_postpro_filename=prepro_data_filepaths["step_2"],
                )

            elif args.detector_name == "GDINO_1.5":
                assert len(loader) == 4896
                _ = postpro_step_2_gdino_1_5(
                    args=args,
                    loader=loader,
                    save_postpro_filename=prepro_data_filepaths["step_2"],
                    timing=True,
                )

            elif args.detector_name in [
                "Florence-2-large-v2",
                "Florence-2-large-ft-v2",
            ]:
                _ = postpro_step_2_florence2(
                    args=args,
                    loader=loader,
                    save_postpro_filename=prepro_data_filepaths["step_2"],
                    timing=True,
                )

            elif args.detector_name == "KOSMOS":
                _ = postpro_step_2_KOSMOS(
                    args=args,
                    loader=loader,
                    save_postpro_filename=prepro_data_filepaths["step_2"],
                    timing=True,
                    micro_timing=False,
                    verbose=False,
                )

    else:
        print(f"VLM preprocessing with {args.detector_name} already done")

    return None


if __name__ == "__main__":

    args = get_args()

    main(args)
