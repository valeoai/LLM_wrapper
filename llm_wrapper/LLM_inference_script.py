import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, box_iou
from tqdm import tqdm

from llm_wrapper.config_llm_wrapper import get_args
from llm_wrapper.LLM_utils import (
    extract_int_result_from_LLM_gen,
    gen_step_3_with_HF,
    get_HF_model_id,
    load_FT_model_via_base_model,
    load_HF_model_tok,
)
from llm_wrapper.naming_utils import (
    define_FT_data_filename,
    define_model_result_name,
    define_short_LLM_folder_name,
    get_checkpoint_list_and_paths_for_eval,
    get_VLM_preprocessed_files_paths,
)
from llm_wrapper.REC_datasets import dataset_for_step_3_various_detectors
from llm_wrapper.utils import convert_box_coord_list

# ----------------------------------------------------------------------
# Useful functions to load preprocessed data (by off-the-shelf VLMs)
# ----------------------------------------------------------------------


def get_box_key_list(detector_name, prompt_context):

    if detector_name == "KOSMOS":
        tres_box_key_list = ["REC", "grounding"]

    elif detector_name in ["Florence-2-large-v2", "Florence-2-large-ft-v2"]:
        # Setting used for main paper results
        if prompt_context == 18:
            tres_box_key_list = ["ORI_QUERY_REC_like", "ORI_QUERY_GROUNDING"]
        # Other good option with slightly better results in most cases as more boxes inferred
        elif prompt_context == 20:
            tres_box_key_list = [
                "ORI_QUERY_REC_like",
                "ORI_QUERY_GROUNDING",
                "DENSE_REGION_CAPTION",
            ]
        else:
            raise Exception("this prompt context end not implemented for Florence-2")

    elif detector_name in ["GDINO_SwinT_original", "GDINO_SwinB_original"]:
        tres_box_key_list = ["ori_thresholded_boxes"]

    elif detector_name in ["GDINO_SwinT", "GDINO_SwinB"]:
        tres_box_key_list = ["thresholded_boxes_for_specific_tok_scores"]

    else:
        raise Exception(f"{detector_name} not implemented")

    return tres_box_key_list


def get_all_rec_and_thres_dict_info_concat(args, one_result_from_step_2, count_per_category=False):

    list_of_all_dict_info = []

    if count_per_category:
        gd_gdrec_rec_boxes_count, other_boxes_count = 0, 0

    # first get rec dict info (separate only for GD/GDrec)
    # In the paper's experiments, we include REC boxes as prompt_context for GDrec but not for GD
    if ("GDINO_Swin" in args.detector_name and args.prompt_context == 4) or count_per_category:

        key = "REC_for_subj_tok_scores" if "REC_for_subj_tok_scores" in one_result_from_step_2 else "REC"
        ori_rec_dict = one_result_from_step_2[key]

        if "box" in ori_rec_dict:

            rec_boxes = ori_rec_dict["box"]
            if "score" in ori_rec_dict:
                rec_scores = ori_rec_dict["score"]
            else:
                rec_scores = [0.4 for e in rec_boxes]
            if not isinstance(rec_scores, list):
                rec_scores = rec_scores.tolist()

            if not isinstance(rec_boxes, list):
                if rec_boxes.dim() > 2:
                    rec_boxes = rec_boxes.squeeze(0)

            if not isinstance(rec_boxes, list):
                rec_boxes = rec_boxes.tolist()

            if len(rec_boxes) > 0 and rec_boxes != [] and rec_boxes != [[]]:
                for one_rec_box, one_rec_score in zip(rec_boxes, rec_scores):
                    one_rec_dict = {
                        "query": one_result_from_step_2["original_full_query"][0].lower().strip(),
                        "one_box": [coord for coord in convert_box_coord_list(one_rec_box, out_format=args.bbox_format)],
                        "one_score": one_rec_score,
                    }
                    if args.prompt_context == 4:
                        list_of_all_dict_info.append(one_rec_dict)

                    if count_per_category:
                        gd_gdrec_rec_boxes_count += 1

    # then get all other boxes info (thresholded dict info for GD/GDrec + both REC/grounding boxes for other detectors)
    tres_box_key_list = get_box_key_list(detector_name=args.detector_name, prompt_context=args.prompt_context)

    for tres_box_key in tres_box_key_list:
        if tres_box_key == "REC" and args.detector_name == "KOSMOS":
            if not isinstance(one_result_from_step_2[tres_box_key], list):
                box_dict = one_result_from_step_2[tres_box_key]
                box_dict["nb_box_found"] = torch.tensor(1)
                box_dict["found_boxes"] = box_dict["box"].unsqueeze(0)
                one_result_from_step_2[tres_box_key] = [box_dict]

        for elem in one_result_from_step_2[tres_box_key]:

            if int(elem["nb_box_found"].item()) != 0:
                query = elem["query"][0]  # string
                found_boxes = elem["found_boxes"].squeeze(0).tolist()  # list of nb_found_box lists of 4 coords
                if "found_boxes_scores" in elem:
                    if not isinstance(elem["found_boxes_scores"], list):
                        found_boxes_scores = elem["found_boxes_scores"].squeeze(0).tolist()  # list of nb_found_box scores
                    else:
                        found_boxes_scores = elem["found_boxes_scores"]
                    assert len(found_boxes) == len(found_boxes_scores)
                else:
                    found_boxes_scores = [0.33 for e in found_boxes]
                if "found_boxes_scores_per_token" in elem:
                    found_boxes_scores_per_token = (
                        elem["found_boxes_scores_per_token"].squeeze(0).tolist()
                    )  # list of nb_found_box lists of nb_tokens token scores

                for it, (one_box, one_score) in enumerate(zip(found_boxes, found_boxes_scores)):
                    one_dict = {
                        "query": query.lower().strip(),
                        "one_box": [round(coord, 1) for coord in convert_box_coord_list(one_box, out_format=args.bbox_format)],
                        "one_score": one_score,
                    }
                    if "found_boxes_scores_per_token" in elem:
                        one_dict["scores_per_token"] = found_boxes_scores_per_token[it]  # list of nb_tokens token scores
                        one_dict["token_list"] = [
                            e[0] for e in elem["noun_group_token_str_list"]
                        ]  # list of nb_tokens token strings

                    list_of_all_dict_info.append(one_dict)

                    if count_per_category:
                        other_boxes_count += 1

    # Truncate the total number of candidate boxes if specified (by default, we let all boxes)
    if args.max_nb_of_box_proposal is not None:
        list_of_all_dict_info = list_of_all_dict_info[: args.max_nb_of_box_proposal]

    if count_per_category:
        count_per_category_dict = {
            "gd_gdrec_rec_boxes_count": gd_gdrec_rec_boxes_count,
            "other_boxes_count": other_boxes_count,
        }
        return list_of_all_dict_info, count_per_category_dict

    return list_of_all_dict_info


def get_all_rec_and_thres_boxes_concat(args, one_result_from_step_2):

    candidate_boxes = torch.empty((0, 4))

    # first get rec dict info (separate only for GD/GDrec), only usded for GDrec in paper results
    if "GDINO_Swin" in args.detector_name and args.prompt_context == 4:
        key = "REC_for_subj_tok_scores" if "REC_for_subj_tok_scores" in one_result_from_step_2 else "REC"
        rec_boxes = one_result_from_step_2[key]["box"]
        if rec_boxes.dim() > 2:
            rec_boxes = rec_boxes.squeeze(0)
        candidate_boxes = torch.cat((candidate_boxes, rec_boxes), dim=0)

    # then get all other boxes info
    tres_box_key_list = get_box_key_list(detector_name=args.detector_name, prompt_context=args.prompt_context)

    for tres_box_key in tres_box_key_list:

        if tres_box_key == "REC" and args.detector_name == "KOSMOS":
            if not isinstance(one_result_from_step_2[tres_box_key], list):
                box_dict = one_result_from_step_2[tres_box_key]
                box_dict["nb_box_found"] = torch.tensor(1)
                box_dict["found_boxes"] = box_dict["box"].unsqueeze(0)
                one_result_from_step_2[tres_box_key] = [box_dict]

        for elem in one_result_from_step_2[tres_box_key]:
            if int(elem["nb_box_found"].item()) != 0:
                candidate_boxes = torch.cat((candidate_boxes, elem["found_boxes"].squeeze(0)), dim=0)  # cxcywh

    # Truncate the total number of candidate boxes if specified (by default, we let all boxes)
    if args.max_nb_of_box_proposal is not None:
        candidate_boxes = candidate_boxes[: args.max_nb_of_box_proposal, :]

    return candidate_boxes


# ----------------------------------------------------------------------
# Useful functions to create input prompts for the LLM(s)
# ----------------------------------------------------------------------


def design_prompt_for_step_3(
    args,
    one_result_from_step_2,
    list_of_all_dict_info=None,
    fixed_seed=None,
    box_count_start=0,
    this_detector=None,
    start_prompt=True,
    end_prompt=True,
    use_prompt_v2=False,
    verbose=False,
):

    if use_prompt_v2:
        prompt_version = args.prompt_version_2
    else:
        prompt_version = args.prompt_version

    if this_detector is not None:
        if "Florence" in this_detector:
            this_detector = "Florence-2"
        elif "GDINO" in this_detector:
            this_detector = "Grounding DINO"
        else:
            raise Exception(f"not implemented for {this_detector}")

    # Get all found boxes
    if list_of_all_dict_info is None:
        list_of_all_dict_info = get_all_rec_and_thres_dict_info_concat(
            args=args, one_result_from_step_2=one_result_from_step_2
        )
    nb_candidate_boxes = len(list_of_all_dict_info)

    box_idx_list = list(np.arange(nb_candidate_boxes))
    if fixed_seed is not None:
        random.seed(fixed_seed)
        random.shuffle(box_idx_list)

    # Build box list as 'context'
    context = ""
    for box_count, idx in enumerate(box_idx_list):

        result_dict = list_of_all_dict_info[idx]
        query = result_dict["query"]
        one_box = result_dict["one_box"]
        one_box = [round(coord, 1) for coord in one_box]

        # listing only coordinates
        if prompt_version == 31:
            if args.detector_name_bis is None:
                context += (
                    f"\n* In box {box_count + box_count_start}: '{query}' with {args.bbox_format} coordinates '{one_box}',"
                )
            else:
                context += (
                    f"\n* Box {box_count + box_count_start}: {this_detector} found '{query}' "
                    f"with {args.bbox_format} coordinates '{one_box}',"
                )

        # listing only coordinates and relevance scores
        elif prompt_version == 32:
            one_score = (
                round(result_dict["one_score"].item(), 2)
                if torch.is_tensor(result_dict["one_score"])
                else round(float(result_dict["one_score"]), 2)
            )
            if args.detector_name_bis is None:
                context += (
                    f"\n* In box {box_count + box_count_start}: '{query}' "
                    "with {args.bbox_format} coordinates '{one_box}' with score {one_score},"
                )
            else:
                context += (
                    f"\n* Box {box_count + box_count_start}: {this_detector} found '{query}' "
                    f"with {args.bbox_format} coordinates '{one_box}' with score {one_score},"
                )

        else:
            raise Exception("Not implemented")

    context = context.strip(",") + "."

    # Building full prompt
    original_full_query = one_result_from_step_2["original_full_query"][0]
    if args.detector_name_bis is None:
        sys_message = (
            "You are a helpful AI assistant, capable of understanding spatial information."
            f"\nIn an image, there are {nb_candidate_boxes} boxes:"
        )
    else:
        sys_message = (
            "You are a helpful AI assistant, capable of understanding spatial information."
            "\nIn an image, I got boxes from two object detectors:"
        )

    user_question = (
        f"\nWhich box is best matching '{original_full_query}' ? Answer with just the index of the best box. No explanation."
    )

    prompt = sys_message if start_prompt else ""
    prompt += context
    if end_prompt:
        prompt += user_question

    # At the end set right template depending on the LLM used
    if "Llama" in args.HF_LLM_name or "llama" in args.HF_LLM_name:
        if start_prompt:
            prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt}"""
        if end_prompt:
            prompt += """<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    elif "Mixtral" in args.HF_LLM_name:
        if start_prompt:
            prompt = f"<s> [INST] {prompt}"
        if end_prompt:
            prompt += "\nAnswer: [/INST] "
    elif "gemma" in args.HF_LLM_name:
        if start_prompt:
            prompt = f"<bos><start_of_turn>user\n {prompt}"
        if end_prompt:
            prompt += "<end_of_turn>\n<start_of_turn>model\n"
    elif "gpt-neo" in args.HF_LLM_name:
        if start_prompt:
            prompt = f"{prompt}"
        if end_prompt:
            prompt += "\nAnswer:"
    else:
        raise Exception(f"prompt template not implemented for LLM {args.HF_LLM_name}")

    if verbose:
        print("\n\nPrompt created:", prompt)

    return prompt, box_idx_list


# ---------------------------------------------------------------------------------------------------
# CASE 1: CREATE FINE-TUNING DATASET : input (text prompt) & output (ideal completion or "label")
# ---------------------------------------------------------------------------------------------------


def create_prompt_label_FT_dataset(args, loader_step_3, use_tqdm=False, timing=True):

    if timing:
        s = time.time()

    with open(args.HF_token_path) as f:
        HF_TOKEN = json.load(f)

    if args.viz_some_FT_data and args.dataset in [
        "RefCOCOg_umd",
        "RefCOCO_unc",
        "RefCOCO+_unc",
    ]:
        from utils import visu_refcocog

        sys.path.append(os.path.join("../", "refer"))
        from refer import REFER

        refer = REFER("../refer/data", "refcocog", "umd")

    # define save path
    short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=args.HF_LLM_name)
    this_detector = args.detector_name if args.detector_name_bis is None else args.detector_name + "_" + args.detector_name_bis
    root_folder = os.path.join(
        args.main_folder,
        args.dataset,
        this_detector,
        short_LLM_name,
        "prompt_dataset_for_finetune",
    )
    save_path = os.path.join(root_folder, args.FT_data_modif)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    filename = define_FT_data_filename(args=args, max_samples=args.max_samples)

    if args.viz_some_FT_data and args.dataset in [
        "RefCOCOg_umd",
        "RefCOCO_unc",
        "RefCOCO+_unc",
    ]:
        clean_image_save_folder = os.path.join(save_path, "viz_images", filename, "good_iou")
        unclean_image_save_folder = os.path.join(save_path, "viz_images", filename, "bad_iou")
        for image_save_folder in [clean_image_save_folder, unclean_image_save_folder]:
            if not os.path.exists(image_save_folder):
                os.makedirs(image_save_folder, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer

    model_id, base_model_name = get_HF_model_id(HF_LLM_name=args.HF_LLM_name)
    cache_dir_tok = os.path.join(args.main_folder, "LLM_cache", f"cache_{base_model_name}_tokenizer")
    print(f"(Down)loading {model_id}'s tokenizer from/to {args.main_folder}/LLM_cache/")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir_tok, token=HF_TOKEN, force_download=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_token = 1024

    # counters for exceptions / nb of LLM inferences
    # c_exceed_tokens, c_too_few_queries, c_single_box = 0, 0, 0
    c_exceed_tokens, c_single_box = 0, 0
    c_llm_pipe_infer = 0

    # Init lists to fill
    all_prompt_dicts_list, CLEAN_prompt_dicts_list = [], []

    pbar = tqdm(enumerate(loader_step_3)) if use_tqdm else enumerate(loader_step_3)
    for it, both_result_from_step_2 in pbar:
        if use_tqdm:
            pbar.set_description(f"Iter {it}")
        if args.detector_name_bis is not None:
            one_result_from_step_2, one_result_from_step_2_bis = both_result_from_step_2
        else:
            one_result_from_step_2 = both_result_from_step_2
            one_result_from_step_2_bis = None
        if it % 1000 == 0:
            print(f"Iter {it}")

        # get all found boxes for 1st VLM
        candidate_boxes = get_all_rec_and_thres_boxes_concat(
            args=args, one_result_from_step_2=one_result_from_step_2
        )  # cxcywh format

        if args.detector_name_bis is not None:
            # get all found boxes for 2nd VLM if specified
            candidate_boxes_bis = get_all_rec_and_thres_boxes_concat(
                args=args, one_result_from_step_2=one_result_from_step_2_bis
            )  # cxcywh format

        nb_candidate_boxes = len(candidate_boxes)
        if args.detector_name_bis is not None:
            nb_candidate_boxes += len(candidate_boxes_bis)

        # -------------------------------------------------------
        # CASE WHERE WE DONT PROCESS DATA
        # -------------------------------------------------------

        if nb_candidate_boxes <= 1:
            all_prompt_dicts_list.append({})
            CLEAN_prompt_dicts_list.append({})
            c_single_box += 1
            continue

        # -------------------------------------------------------
        # CASE WHERE WE PROCESS DATA
        # -------------------------------------------------------
        # Getting the prompt

        result_prompts_dict = {}

        if args.detector_name_bis is None:
            prompt_for_step_3, box_idx_list = design_prompt_for_step_3(
                args=args,
                one_result_from_step_2=one_result_from_step_2,
                fixed_seed=None,
            )

        else:
            prompt_for_step_3, box_idx_list = design_prompt_for_step_3(
                args=args,
                one_result_from_step_2=one_result_from_step_2,
                box_count_start=0,
                this_detector=args.detector_name,
                start_prompt=True,
                end_prompt=False,
            )

            prompt_for_step_3_bis, box_idx_list_bis = design_prompt_for_step_3(
                args=args,
                one_result_from_step_2=one_result_from_step_2_bis,
                box_count_start=len(candidate_boxes),
                this_detector=args.detector_name_bis,
                start_prompt=False,
                end_prompt=True,
                use_prompt_v2=True,
            )

            prompt_for_step_3 = prompt_for_step_3 + prompt_for_step_3_bis

        encoded_input = tokenizer(prompt_for_step_3, return_tensors="pt", add_special_tokens=False)
        nb_tok = encoded_input["input_ids"].shape[-1]

        if nb_tok > max_token:
            c_exceed_tokens += 1
            all_prompt_dicts_list.append({})
            CLEAN_prompt_dicts_list.append({})
            print("too many tokens")
            continue

        c_llm_pipe_infer += 1

        result_prompts_dict["None"] = prompt_for_step_3
        if it <= 2 and not args.verbose:
            print(f"\n\nPrompt (v {args.prompt_version}):\n", prompt_for_step_3)

        # Get seed list for augmentation if requested
        # NOT IMPLEMENTED FOR 2 VLMS
        if args.FT_data_modif == "aug":
            if nb_candidate_boxes == 1:
                seed_list = []
            if nb_candidate_boxes == 2:
                seed_list = [2]
            elif nb_candidate_boxes == 3:
                seed_list = [2, 6]
            elif nb_candidate_boxes == 4:
                seed_list = [0, 2, 7]
            else:
                seed_list = [2, 4, 6, 7]
        elif args.FT_data_modif == "shuffle":
            seed_list = [random.randint(0, 9)]
        else:  # "no_modif"
            seed_list = []

        for seed in seed_list:

            prompt_for_step_3, box_idx_list = design_prompt_for_step_3(
                args=args,
                one_result_from_step_2=one_result_from_step_2,
                fixed_seed=seed,
            )
            result_prompts_dict[seed] = prompt_for_step_3

        # Getting subject candidates boxes and label (the index that should be generated)

        if args.detector_name_bis is not None:
            candidate_boxes = torch.cat((candidate_boxes, candidate_boxes_bis), dim=0)

        one_rec_result, one_iou, result_idx_dict = eval_and_find_best_subj(
            infer_bbox=candidate_boxes,
            one_result_from_step_2=one_result_from_step_2,
            bbox_format=args.bbox_format,
            prompt_version=args.prompt_version,
            seed_list=seed_list,
            verbose=args.verbose,
        )  # 1 if found, 0 otherwise

        if args.viz_some_FT_data and it < 10 and args.dataset in ["RefCOCOg_umd", "RefCOCO_unc", "RefCOCO+_unc"]:

            image_id = one_result_from_step_2["image_id"].item()
            category_id = one_result_from_step_2["category_id"].item()
            img_filename = f"img_id_{image_id}_category_id_{category_id}.png"

            original_full_query = str(one_result_from_step_2["original_full_query"][0])
            labelled_box = candidate_boxes[result_idx_dict["None"]].unsqueeze(0)  # torch.Size([1, 4])
            labelled_box = box_convert(labelled_box, in_fmt="cxcywh", out_fmt="xyxy").squeeze(0).tolist()  # list of len 4
            GT_box = one_result_from_step_2["GT_bbox"][0]  # format xyxy, original image scale

            viz_save_filepath = (
                os.path.join(clean_image_save_folder, img_filename)
                if one_rec_result == 1
                else os.path.join(unclean_image_save_folder, img_filename)
            )
            visu_refcocog(
                refer=refer,
                main_folder=args.main_folder,
                image_id=image_id,
                text_no_modif=f"labeled_box (iou={round(one_iou,2)})",
                text_modif=None,
                box_no_modif=labelled_box,
                box_modif=None,
                box_GT=GT_box,
                box_GT_format="xyxy",
                title_text=original_full_query,
                save_filepath=viz_save_filepath,
            )

        # Appending prompt / label to 2 lists (full/clean)

        if args.FT_data_modif == "aug":
            seed_list = ["None"] + seed_list
        elif args.FT_data_modif == "shuffle":
            seed_list = seed_list
        else:  # "no_modif"
            seed_list = ["None"]

        for seed_key in seed_list:

            prompt_for_seed = result_prompts_dict[seed_key]
            best_subj_idx_for_seed = result_idx_dict[seed_key]

            all_prompt_dicts_list.append(
                {
                    "prompt": prompt_for_seed,
                    "best_subject_index": str(best_subj_idx_for_seed),
                    "iou_best_subj_to_GT": round(one_iou, 3),
                }
            )
            if one_iou > 0.5:
                CLEAN_prompt_dicts_list.append(
                    {
                        "prompt": prompt_for_seed,
                        "best_subject_index": str(best_subj_idx_for_seed),
                        "iou_best_subj_to_GT": round(one_iou, 3),
                    }
                )

    # -------------------------------------------------------
    # Stats printed

    print("\nTotal nb of LLM inferences:", c_llm_pipe_infer)
    print("\nTotal nb of exceptions:")
    print("Nb of prompts exceeding the max input token limit:", c_exceed_tokens)
    print("Nb of cases where 0 or 1 candidate box found only:", c_single_box)

    # -------------------------------------------------------
    # Dumping two built dict lists

    with open(os.path.join(save_path, filename + ".json"), "w") as outfile:
        json.dump(all_prompt_dicts_list, outfile, indent=4)

    with open(os.path.join(save_path, filename + "_CLEAN.json"), "w") as outfile:
        json.dump(CLEAN_prompt_dicts_list, outfile, indent=4)

    if timing:
        e = time.time()
        print("\n\nPseudo step 3 time:", round(e - s, 1), "secs\n")

    return None


def eval_and_find_best_subj(
    infer_bbox,
    one_result_from_step_2,
    bbox_format,
    prompt_version,
    seed_list=[],
    verbose=False,
):
    # one_result_from_step_2['GT_bbox'] is a list or an array with "xyxy" format => required for box_iou
    # infer_bbox is a tensor of dim (nb_subj_candidates, 4) and is in bbox_format

    result_idx_dict = {}

    # case when there are 0 or 1 candidate box
    nb_subj_candidates = infer_bbox.shape[0]
    if nb_subj_candidates == 0:
        raise Exception("WARNING NO CANDIDATE BOXES, should not happen")
    if nb_subj_candidates == 1:
        print("WARNING ONLY 1 CANDIDATE BOX")

    # ensure both box objects are tensors with xyxy format
    infer_bbox = box_convert(infer_bbox, in_fmt="cxcywh", out_fmt="xyxy")

    if verbose:
        print("GT:", one_result_from_step_2["GT_bbox"])
        print("infer_bbox:", infer_bbox)

    if isinstance(one_result_from_step_2["GT_bbox"], list):
        GT_bbox = torch.tensor(one_result_from_step_2["GT_bbox"])
    else:
        GT_bbox = one_result_from_step_2["GT_bbox"]

    # iou / metric computation
    iou_matrix = box_iou(GT_bbox, infer_bbox).squeeze(0)  # dim (nb_subj_candidates)

    # find index of best subject box (for no shuffle of box order)
    best_subject_index = torch.argmax(iou_matrix).item()  # int
    result_idx_dict["None"] = best_subject_index
    max_iou = torch.max(iou_matrix).item()
    one_rec_result = 1 if max_iou > 0.5 else 0

    if verbose:
        print("iou_matrix:", iou_matrix)
        print("max_iou:", round(max_iou, 3))
        print("best_subject_index:", best_subject_index)
        print("rec result=", one_rec_result, f"(iou={round(max_iou,3)}) \n")

    if len(seed_list) > 0:
        ori_idx = list(np.arange(nb_subj_candidates))
        for seed in seed_list:
            idx = ori_idx.copy()
            random.seed(seed)
            random.shuffle(idx)
            new_best_subject_index = idx.index(best_subject_index)
            result_idx_dict[seed] = new_best_subject_index

    return one_rec_result, max_iou, result_idx_dict


# ------------------------------------------------
# CASE 2: LLM-WRAPPER INFERENCE
# ------------------------------------------------


def unify_output_step_3(
    result_list,
    one_result_from_step_2,
    one_result_from_step_2_bis=None,
    prompt_version=None,
    bbox_format=None,
    return_tensor=True,
):
    # boxes in result_list are already converted to bbox format
    # if no valid answer from LLM, fallback on REC result from VLM

    if len(result_list) == 0:
        key = "REC_for_subj_tok_scores" if "REC_for_subj_tok_scores" in one_result_from_step_2 else "REC"
        if isinstance(one_result_from_step_2[key], list):  # and args.detector_name == "KOSMOS":
            rec_boxes = one_result_from_step_2[key][0]["box"]
        else:
            rec_boxes = one_result_from_step_2[key]["box"]

        if rec_boxes in [
            [],
            [[]],
        ]:  # case where no rec preds especially for Flo2 FT on some datasets
            infer_bbox = [0, 0, 0, 0]

        else:
            if isinstance(rec_boxes, list):
                if isinstance(rec_boxes[0], list):
                    infer_bbox = rec_boxes[0]
                else:
                    infer_bbox = rec_boxes
                print("infer_bbox for debug:", infer_bbox)
            elif rec_boxes.dim() > 2:
                rec_boxes = rec_boxes.squeeze(0)
            infer_bbox = rec_boxes[0]

    else:

        if len(result_list) == 1:
            subject_idx = result_list[0]
        else:
            subject_idx = Counter(result_list).most_common()[0][0]

        # get all candidate boxes to retrieve the chosen one
        found_boxes = get_all_rec_and_thres_boxes_concat(args=args, one_result_from_step_2=one_result_from_step_2)
        if one_result_from_step_2_bis is not None:
            found_boxes_bis = get_all_rec_and_thres_boxes_concat(args=args, one_result_from_step_2=one_result_from_step_2_bis)
            found_boxes = torch.cat((found_boxes, found_boxes_bis), dim=0)

        infer_bbox = found_boxes[subject_idx]

    infer_bbox = convert_box_coord_list(infer_bbox, out_format=bbox_format)

    # make a [1,4] dim tensor if needed from list
    if return_tensor:
        infer_bbox = torch.tensor(infer_bbox).unsqueeze(0)

    return infer_bbox


def eval_post_step_3(infer_bbox, one_result_from_step_2, bbox_format, verbose=False):
    # one_result_from_step_2['GT_bbox'] is a tensor with "xyxy" format => required for box_iou
    # infer_bbox is a tensor of dim (1,4) (or (nb_candidats, 4) when using oracle prompt 0) and is in bbox_format

    # case when there are no candidate boxes
    if infer_bbox.shape[0] == 0:
        one_rec_result, max_iou = 0, 0

    else:
        # ensure both box objects are tensors with xyxy format
        if bbox_format != "xyxy":
            infer_bbox = box_convert(infer_bbox, in_fmt=bbox_format, out_fmt="xyxy")
        if verbose:
            print("GT:", one_result_from_step_2["GT_bbox"])
        GT_bbox = one_result_from_step_2["GT_bbox"]

        # iou / metric computation
        iou_matrix = box_iou(GT_bbox, infer_bbox).squeeze(0)  # dim (infer_bbox.shape[0])

        max_iou = torch.max(iou_matrix).item()
        one_rec_result = 1 if max_iou > 0.5 else 0
        if verbose:
            print("rec result=", one_rec_result, f"(iou={round(max_iou,3)}) \n")

    return one_rec_result, max_iou


# 'step 3'
def REC_eval_step(
    args,
    loader_step_3,
    llm,
    tokenizer,
    print_error_stats=False,
    timing=True,
    manual_dont_save=False,
    use_tqdm=True,
    checkpoint_nb=None,
    folder_with_checkpoints=None,
    with_explanation=False,
):

    if timing:
        s = time.time()

    if args.VLM_only:
        actual_rec_box, it_list_wo_rec_preds = 0, []
        no_preds, use_grounding_instead_of_rec = 0, 0
    else:
        it_list_wo_rec_preds = None

    if args.prompt_version != 0 and not args.VLM_only:

        if llm is None:
            llm, tokenizer = load_HF_model_tok(
                args=args,
                HF_LLM_name=args.HF_LLM_name,
                eval_mode=True,
                FT_mode=False,
                timing=True,
            )
        else:
            print("Use given LLM (no init)")

    min_nb_candidates, max_nb_candidates, avg_nb_candidates, nb_oracle_it = 200, 0, 0, 0
    avg_iou, nb_found_boxes, c_nb_eval = 0, 0, 0
    c_nb_failed_box_inference = 0
    c_llm_pipe_infer = 0
    (
        c_0_candidates,
        c_1_candidates,
        c_failed_to_gen,
        c_gen_out_of_scope,
        c_no_int_gen,
    ) = (0, 0, 0, 0, 0)
    all_infer_bboxes = []

    if not args.dont_save_results:
        correct_indices, LLM_infer_indices = [], []

    if args.limit_to_300_samples_GDINO_1_5:
        if args.dataset != "RefCOCOg_umd" or args.split != "val":
            raise Exception(f"cannot use limit_to_300_samples_GDINO_1_5 with {args.dataset} or {args.split}")
        sample_id_list_path = (
            "/home/acardiel/scania/store_scania/acardiel/RefCOCOg_umd/GDINO_1.5/list_of_300_samples_used.json"
        )
        with open(sample_id_list_path) as f:
            sample_id_list = json.load(f)
        c_300_samples = 0

    # Start loop

    pbar = tqdm(enumerate(loader_step_3)) if use_tqdm else enumerate(loader_step_3)
    for it, both_result_from_step_2 in pbar:
        if use_tqdm:
            pbar.set_description(f"Iter {it}")
        if args.detector_name_bis is not None:
            one_result_from_step_2, one_result_from_step_2_bis = both_result_from_step_2
        else:
            one_result_from_step_2 = both_result_from_step_2
            one_result_from_step_2_bis = None
        if args.limit_to_300_samples_GDINO_1_5:
            image_id = one_result_from_step_2["image_id"].item()
            sent_id = one_result_from_step_2["category_id"].item()
            print("image_id:", image_id, " | sent_id:", sent_id)
            sample_id = {"img_id": image_id, "sent_id": sent_id}
            if sample_id not in sample_id_list:
                continue
            else:
                c_300_samples += 1
                print(c_300_samples, "sample found")

        ###
        # 1) Select a box to eval for REC (infer_bbox)
        ###

        # 1-a) If want to use VLM only instead of LLM-wrapper

        if args.VLM_only:

            if args.detector_name not in ["GDINO_SwinT", "GDINO_SwinB"]:
                if one_result_from_step_2["REC"]["box"] in [
                    [],
                    [[]],
                ]:  # if no REC box found for this it

                    # specific to flo2(FT) for experiments only
                    if args.spe_fall_back_for_vlm_0s and ("ORI_QUERY_GROUNDING" in one_result_from_step_2):

                        if int(one_result_from_step_2["ORI_QUERY_GROUNDING"][0]["nb_box_found"]) > 0:
                            use_grounding_instead_of_rec += 1
                            result = one_result_from_step_2["ORI_QUERY_GROUNDING"][0]["found_boxes"].squeeze(0)[0]
                            result = convert_box_coord_list(result, out_format=args.bbox_format)
                            infer_bbox = torch.tensor(result).unsqueeze(0)
                        else:
                            no_preds += 1
                            infer_bbox = torch.zeros((1, 4))  # wrong box on purpose
                            it_list_wo_rec_preds.append(it)

                    # default in all expe in paper results
                    else:
                        no_preds += 1
                        infer_bbox = torch.zeros((1, 4))  # wrong box on purpose
                        it_list_wo_rec_preds.append(it)

                else:
                    actual_rec_box += 1
                    result = one_result_from_step_2["REC"]["box"].squeeze(0)
                    # taking only first REC box in case florence-2 generated more than 1
                    if result.dim() > 1:
                        result = result[0]
                    result = convert_box_coord_list(result, out_format=args.bbox_format)
                    infer_bbox = torch.tensor(result).unsqueeze(0)

            elif args.detector_name in ["GDINO_SwinT", "GDINO_SwinB"]:
                rec_key = "REC_for_subj_tok_scores" if "box" in one_result_from_step_2["REC_for_subj_tok_scores"] else "REC"
                actual_rec_box += 1
                result = one_result_from_step_2[rec_key]["box"].squeeze(0)
                result = convert_box_coord_list(result, out_format=args.bbox_format)
                infer_bbox = torch.tensor(result).unsqueeze(0)

        # 1-b) If want to use an Oracle for LLM-wrapper (if one could choose the best box among candidates)

        elif args.prompt_version == 0:

            if args.detector_name_bis is not None:
                raise Exception("Oracle not implemented when use ensembling of 2 VLM boxes")

            # Get all candidate boxes from step 2 (cxcywh format, original image scale)
            infer_bbox = get_all_rec_and_thres_boxes_concat(
                args, one_result_from_step_2
            )  # tensor of dim (nb_candidats, 4), cxcywh format

            nb_candidate_boxes = int(infer_bbox.shape[0])
            if nb_candidate_boxes > max_nb_candidates:
                max_nb_candidates = nb_candidate_boxes
            if nb_candidate_boxes < min_nb_candidates:
                min_nb_candidates = nb_candidate_boxes
            avg_nb_candidates += nb_candidate_boxes
            nb_oracle_it += 1

            if nb_candidate_boxes == 0:
                c_nb_failed_box_inference += 1
            else:
                infer_bbox = box_convert(infer_bbox, in_fmt="cxcywh", out_fmt=args.bbox_format)

        # 1-c) Use LLM-wrapper

        else:

            found_box = False

            # Retrieve VLM detections for our chosen VLM(s)

            tres_box_key_list = get_box_key_list(detector_name=args.detector_name, prompt_context=args.prompt_context)
            all_subqueries_boxes_scores_from_step_2 = []
            for tres_box_key in tres_box_key_list:
                all_subqueries_boxes_scores_from_step_2.extend(one_result_from_step_2[tres_box_key])
            list_of_all_dict_info, count_per_category_dict = get_all_rec_and_thres_dict_info_concat(
                args=args,
                one_result_from_step_2=one_result_from_step_2,
                count_per_category=True,
            )
            nb_candidate_boxes = len(list_of_all_dict_info)

            if args.detector_name_bis is not None:
                tres_box_key_list_bis = get_box_key_list(
                    detector_name=args.detector_name_bis,
                    prompt_context=args.prompt_context_2,
                )
                all_subqueries_boxes_scores_from_step_2_bis = []
                for tres_box_key in tres_box_key_list_bis:
                    all_subqueries_boxes_scores_from_step_2_bis.extend(one_result_from_step_2_bis[tres_box_key])
                list_of_all_dict_info_bis, count_per_category_dict_bis = get_all_rec_and_thres_dict_info_concat(
                    args=args,
                    one_result_from_step_2=one_result_from_step_2_bis,
                    count_per_category=True,
                )
                nb_candidate_boxes += len(list_of_all_dict_info_bis)

            # Exception: if only 0 or 1 box candidate, we do not go through the LLM pipeline

            if nb_candidate_boxes == 0:
                found_box = True
                c_0_candidates += 1
                infer_bbox = torch.zeros((1, 4))

            if nb_candidate_boxes == 1:
                found_box = True
                c_1_candidates += 1

                # if single box comes from 1st VLM
                if len(list_of_all_dict_info) > 0:
                    count_dict = count_per_category_dict
                    used_detector = args.detector_name
                    res_step_2 = one_result_from_step_2
                    all_subqueries_step_2 = all_subqueries_boxes_scores_from_step_2

                # if single box comes from 2nd VLM
                else:
                    count_dict = count_per_category_dict_bis
                    used_detector = args.detector_name_bis
                    res_step_2 = one_result_from_step_2_bis
                    all_subqueries_step_2 = all_subqueries_boxes_scores_from_step_2_bis

                # retrieve the unique detected box
                if count_dict["gd_gdrec_rec_boxes_count"] > 0 or used_detector == "GDINO_1.5":
                    key = "REC_for_subj_tok_scores" if "REC_for_subj_tok_scores" in res_step_2 else "REC"
                    rec_boxes = res_step_2[key]["box"]
                    if isinstance(rec_boxes, list):
                        rec_boxes = torch.tensor(rec_boxes)
                    if rec_boxes.dim() > 2:
                        rec_boxes = rec_boxes.squeeze(0)
                    infer_bbox = rec_boxes[0].unsqueeze(0)
                    infer_bbox = box_convert(infer_bbox, in_fmt="cxcywh", out_fmt=args.bbox_format)

                elif count_dict["other_boxes_count"] > 0:

                    for elem in all_subqueries_step_2:
                        if int(elem["nb_box_found"].item()) != 0:
                            found_boxes = elem["found_boxes"].squeeze(0)  # tensor of nb_found_box lists of 4 coords
                            if found_boxes.dim() > 2:
                                found_boxes = found_boxes.squeeze(0)
                            infer_bbox = found_boxes[0].unsqueeze(0)
                            infer_bbox = box_convert(infer_bbox, in_fmt="cxcywh", out_fmt=args.bbox_format)
                            break

            # If no exception => use LLM inference

            if not found_box:

                result_list = []

                for fixed_seed in [None]:

                    # Build textual prompt from VLM detections (for one or two VLMs)

                    # when using a single VLM
                    if args.detector_name_bis is None:
                        prompt_for_step_3, box_idx_list = design_prompt_for_step_3(
                            args=args,
                            one_result_from_step_2=one_result_from_step_2,
                            list_of_all_dict_info=list_of_all_dict_info,
                            fixed_seed=fixed_seed,
                        )

                    # when ensembling 2 VLMs
                    else:
                        prompt_for_step_3, box_idx_list = design_prompt_for_step_3(
                            args=args,
                            one_result_from_step_2=one_result_from_step_2,
                            list_of_all_dict_info=list_of_all_dict_info,
                            box_count_start=0,
                            this_detector=args.detector_name,
                            start_prompt=True,
                            end_prompt=False,
                        )

                        prompt_for_step_3_bis, box_idx_list_bis = design_prompt_for_step_3(
                            args=args,
                            one_result_from_step_2=one_result_from_step_2_bis,
                            list_of_all_dict_info=list_of_all_dict_info_bis,
                            box_count_start=len(list_of_all_dict_info),
                            this_detector=args.detector_name_bis,
                            start_prompt=False,
                            end_prompt=True,
                            use_prompt_v2=True,
                        )

                        prompt_for_step_3 = prompt_for_step_3 + prompt_for_step_3_bis

                    if it <= 2 and args.verbose is False:
                        print(f"\nprompt (v {args.prompt_version}):\n", prompt_for_step_3)

                    # Generate box index with LLM

                    try:
                        result_step_3 = gen_step_3_with_HF(
                            HF_LLM_name=args.HF_LLM_name,
                            model=llm,
                            tokenizer=tokenizer,
                            prompt=prompt_for_step_3,
                            verbose=False,
                            max_gen_tok=args.max_gen_tok,
                        )
                    except:
                        c_failed_to_gen += 1
                        result_step_3 = None

                    # If generation suceeded, try to extract generated int

                    if result_step_3 is not None:

                        nb_detected_boxes = len(box_idx_list)
                        if args.detector_name_bis is not None:
                            nb_detected_boxes += len(box_idx_list_bis)

                        if nb_detected_boxes > max_nb_candidates:
                            max_nb_candidates = nb_detected_boxes
                        if nb_detected_boxes < min_nb_candidates:
                            min_nb_candidates = nb_detected_boxes
                        avg_nb_candidates += nb_detected_boxes
                        nb_oracle_it += 1

                        result_step_3, error_type = extract_int_result_from_LLM_gen(
                            result=result_step_3,
                            HF_LLM_name=args.HF_LLM_name,
                            return_error_type=True,
                            nb_detected_boxes=nb_detected_boxes,
                            fixed_seed=fixed_seed,
                        )

                        if result_step_3 is not None:
                            if it <= 2:
                                print(
                                    "\nExtracted int from LLM generated result:",
                                    result_step_3,
                                )
                            result_list.append(result_step_3)

                        else:
                            if error_type == "gen_out_of_scope":
                                if it <= 2:
                                    print("\nLLM generated int is out of scope")
                                c_gen_out_of_scope += 1
                            elif error_type == "no_int_generated":
                                if it <= 2:
                                    print("\nNo proper int generated by LLM")
                                c_no_int_gen += 1

                if len(result_list) > 0:
                    c_llm_pipe_infer += 1

                    if not args.dont_save_results:
                        LLM_infer_indices.append(it)

                # Retrieve candidate box for REC eval (infer_bbox in bbox_format with dim [1,4])

                infer_bbox = unify_output_step_3(
                    result_list=result_list,
                    one_result_from_step_2=one_result_from_step_2,
                    one_result_from_step_2_bis=one_result_from_step_2_bis,
                    prompt_version=args.prompt_version,
                    bbox_format=args.bbox_format,
                    return_tensor=True,
                )

        ###
        # 2) One REC evaluation
        ###

        # Save coordinates of the chosen box (infer_bbox) if specified

        if args.prompt_version != 0 and not args.dont_save_results:
            box_to_save = [round(coord, 1) for coord in infer_bbox.squeeze(0).detach().cpu().tolist()]
            all_infer_bboxes.append(box_to_save)

        # Compute REC metric and avg iou for the chosen box (infer_bbox)

        one_rec_result, one_iou = eval_post_step_3(
            infer_bbox=infer_bbox,
            one_result_from_step_2=one_result_from_step_2,
            bbox_format=args.bbox_format,
            verbose=args.verbose,
        )  # 1 if found, 0 otherwise
        if not args.dont_save_results and one_rec_result == 1:
            correct_indices.append(it)
        nb_found_boxes += one_rec_result
        avg_iou += one_iou
        c_nb_eval += 1

    ###
    # 3) Compute and save evaluation scores on full dataset split
    ###

    # Sanity check
    if args.limit_to_300_samples_GDINO_1_5:
        print(
            "Number of time we processed the samples as corresponding to those used for GDINO 1.5 eval:",
            c_300_samples,
        )

    # Main metrics (compute and print)
    REC_metric = nb_found_boxes / c_nb_eval if c_nb_eval != 0 else 0
    avg_iou = avg_iou / c_nb_eval if c_nb_eval != 0 else 0
    print(f"\nREC metric (%) on {c_nb_eval} samples:", round(REC_metric * 100, 2))
    print(f"Avg IOU (%) on {c_nb_eval} samples:", round(avg_iou * 100, 2))

    # Minor metrics (compute and print)
    if not args.VLM_only and nb_oracle_it != 0:
        avg_nb_candidates = avg_nb_candidates / nb_oracle_it
        stats_nb_candidate_boxes = {
            "min_nb_candidates": min_nb_candidates,
            "max_nb_candidates": max_nb_candidates,
            "avg_nb_candidates": avg_nb_candidates,
        }
        print("avg_nb_candidates:", avg_nb_candidates)
        print("max_nb_candidates:", max_nb_candidates)

        inference_types = {
            # when no error
            "c_llm_pipe_infer": c_llm_pipe_infer,
            # when error
            "c_0_candidates": c_0_candidates,
            "c_1_candidates": c_1_candidates,
            "c_failed_to_gen": c_failed_to_gen,
            "c_no_int_gen": c_no_int_gen,
            "c_gen_out_of_scope": c_gen_out_of_scope,
        }

        print("Type of LLM inferences:\n", inference_types)
        print("Number of actual LLM inferences in total:", c_llm_pipe_infer)

    if args.VLM_only:
        stats_nb_candidate_boxes = None

        if "Florence" in args.detector_name:
            print("Number of REC boxes / Number of fallback on grounding boxes / Number of no pred. boxes")
            print(actual_rec_box, "/", use_grounding_instead_of_rec, "/", no_preds)
            inference_types = {
                "actual_rec_box": actual_rec_box,
                "use_grounding_instead_of_rec": use_grounding_instead_of_rec,
                "no_preds": no_preds,
            }
        else:
            inference_types = None

    # Saving to files
    if not args.dont_save_results and not manual_dont_save:
        compute_and_save_infer_info(
            REC_metric=REC_metric,
            avg_iou=avg_iou,
            args=args,
            correct_indices=correct_indices,
            LLM_infer_indices=LLM_infer_indices,
            all_infer_bboxes=all_infer_bboxes,
            checkpoint_nb=checkpoint_nb,
            folder_with_checkpoints=folder_with_checkpoints,
            stats_nb_candidate_boxes=stats_nb_candidate_boxes,
            it_list_wo_rec_preds=it_list_wo_rec_preds,
            inference_types=inference_types,
        )

    if timing:
        e = time.time()
        print(
            f"\n\nTime for REC eval on {args.dataset} {args.split}:",
            round(e - s, 1),
            "secs",
        )

    return REC_metric


def compute_and_save_infer_info(
    REC_metric,
    avg_iou,
    args,
    correct_indices,
    LLM_infer_indices,
    all_infer_bboxes,
    checkpoint_nb,
    folder_with_checkpoints,
    stats_nb_candidate_boxes=None,
    it_list_wo_rec_preds=None,
    inference_types=None,
):

    this_detector = args.detector_name if args.detector_name_bis is None else args.detector_name + "_" + args.detector_name_bis

    if args.prompt_version != 0 and not args.VLM_only:
        short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=args.HF_LLM_name)
        save_path = os.path.join(args.main_folder, args.dataset, this_detector, short_LLM_name, "results")
    else:
        save_path = os.path.join(args.main_folder, args.dataset, this_detector, "results_wo_LLM")

    if args.folder_with_checkpoints_to_eval is not None:
        save_path = os.path.join(save_path, folder_with_checkpoints)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    model_name_to_use = define_model_result_name(args, checkpoint_nb)

    index_dict = {
        "REC_metric": REC_metric,
        "avg_iou": avg_iou,
        "nb_LLM_infer": len(LLM_infer_indices),  # should be equal to inference_types's key c_llm_pipe_infer
        "correct_indices": correct_indices,
        "LLM_infer_indices": LLM_infer_indices,
    }

    if it_list_wo_rec_preds is not None:
        index_dict["it_list_wo_rec_preds"] = it_list_wo_rec_preds

    if stats_nb_candidate_boxes is not None:
        index_dict["stats_nb_candidate_boxes"] = stats_nb_candidate_boxes

    if inference_types is not None:
        index_dict["inference_types"] = inference_types

    if len(all_infer_bboxes) != 0:
        index_dict["all_infer_bboxes"] = all_infer_bboxes

    with open(os.path.join(save_path, model_name_to_use), "w") as outfile:
        json.dump(index_dict, outfile, indent=4, sort_keys=True)

    return None


def print_recap_rec_results_in_checkpoint_folder(args, folder_with_checkpoints):
    from utils import print_all_rec_in_folder

    this_detector = args.detector_name if args.detector_name_bis is None else f"{args.detector_name}_{args.detector_name_bis}"
    short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=args.HF_LLM_name)
    save_result_folder = os.path.join(
        args.main_folder,
        args.dataset,
        this_detector,
        short_LLM_name,
        "results",
        folder_with_checkpoints,
    )
    print_all_rec_in_folder(folder_path=save_result_folder)
    return None


# ---------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------


def main(args):

    # 1) GET INFO FROM VLMS

    # Get filepath names of VLM preprocessed data
    prepro_data_filepaths = get_VLM_preprocessed_files_paths(args=args)

    # Check whether VLM preprocessing was done
    if not os.path.exists(prepro_data_filepaths["step_2"]) and not os.path.exists(prepro_data_filepaths["full_step_2"]):
        print(prepro_data_filepaths["step_2"], "does not exist")
        print(prepro_data_filepaths["full_step_2"], "either")
        raise Exception(f"Missing preprocessing of {args.dataset} {args.split} with {args.detector_name}")

    # 2) MAIN PIPELINE : fine-tuning dataset creation or REC evaluation on our chosen dataset & split

    print("\nVLM preprocessing already done => start main pipeline")

    # Get data loader
    if os.path.exists(prepro_data_filepaths["step_2"]):
        dataset_step_3 = dataset_for_step_3_various_detectors(
            filepath=prepro_data_filepaths["step_2"],
            filepath_bis=prepro_data_filepaths["step_2_bis"],
        )
    else:
        dataset_step_3 = dataset_for_step_3_various_detectors(
            filepath=prepro_data_filepaths["full_step_2"],
            filepath_bis=prepro_data_filepaths["full_step_2_bis"],
        )
        if args.detector_name != "GDINO_1.5":
            dataset_step_3 = torch.utils.data.Subset(dataset_step_3, range(args.min_samples, args.max_samples))
    loader_step_3 = DataLoader(
        dataset_step_3,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Dataloader for main pipeline of len {len(dataset_step_3)} created")

    # Option 1: Process VLM outputs to create a prompt & completion fine-tuning dataset
    if args.create_FT_dataset:

        print(
            f"Start building fine-tuning dataset based on {args.max_samples - args.min_samples} samples from {args.split} set"
        )
        create_prompt_label_FT_dataset(args=args, loader_step_3=loader_step_3)

    # Option 2: Get REC eval on our chosen data with "VLM only" or "VLM + LLM-wrapper"
    else:

        print(f"REC performance evaluation on {args.max_samples - args.min_samples} samples from {args.split} set")

        if args.folder_with_checkpoints_to_eval is not None:

            path_to_folder, folder_with_checkpoints, checkpoint_subfolders = get_checkpoint_list_and_paths_for_eval(args=args)

            if len(checkpoint_subfolders) > 0:

                _, base_model_name = get_HF_model_id(HF_LLM_name=folder_with_checkpoints)
                base_model, tokenizer = load_HF_model_tok(
                    args=args,
                    HF_LLM_name=base_model_name,
                    eval_mode=True,
                    FT_mode=False,
                )

                for subfolder in checkpoint_subfolders:
                    print("\nAdd LoRA adapters for", subfolder)
                    FT_model = load_FT_model_via_base_model(
                        base_model=base_model,
                        complete_FT_path=os.path.join(path_to_folder, subfolder),
                        timing=True,
                    )
                    REC_metric = REC_eval_step(
                        args=args,
                        loader_step_3=loader_step_3,
                        llm=FT_model,
                        tokenizer=tokenizer,
                        checkpoint_nb=subfolder,
                        folder_with_checkpoints=folder_with_checkpoints,
                    )

            print_recap_rec_results_in_checkpoint_folder(args=args, folder_with_checkpoints=folder_with_checkpoints)

        else:
            REC_metric = REC_eval_step(args=args, loader_step_3=loader_step_3, llm=None, tokenizer=None)


    return None


if __name__ == "__main__":

    args = get_args()
    main(args)
