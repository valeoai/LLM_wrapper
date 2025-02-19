import os
import time
from collections import Counter

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torchvision.ops import box_convert
from tqdm import tqdm

torch.set_grad_enabled(False)


def load_florence_2(detector_name, main_folder, timing=True):

    if timing:
        s = time.time()

    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = "microsoft/" + detector_name.strip("-v2")
    cache_dir = os.path.join(main_folder, "LLM_cache", f"cache_{model_id}")
    cache_dir_processor = os.path.join(main_folder, "LLM_cache", f"cache_{model_id}_processor")

    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True).eval().cuda()

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir_processor, trust_remote_code=True)

    if timing:
        e = time.time()
        print(f"{detector_name} loading time:", round(e - s, 2), "secs")
    return model, processor


def run_example(model, processor, image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    # return generated_text, parsed_answer
    return parsed_answer


def convert_to_od_format(data):
    """
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.

    Parameters:
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.

    Returns:
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.
    """
    # Extract bounding boxes and labels
    bboxes = data.get("bboxes", [])
    labels = data.get("bboxes_labels", [])

    # Construct the output format
    od_results = {"bboxes": bboxes, "labels": labels}

    return od_results


def plot_bbox(image, box_data, vizu_path):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(box_data["bboxes"], box_data["labels"]):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none")
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(
            x1,
            y1,
            label,
            color="white",
            fontsize=8,
            bbox=dict(facecolor="red", alpha=0.5),
        )

    # Remove the axis ticks and labels
    ax.axis("off")

    # Show the plot
    # plt.show()
    plt.savefig(vizu_path)

    return None


def prepro_step_2_flo2(args, data, timing=True, verbose=False):

    print(f"\nStart preprocessing for {args.detector_name}")

    if timing:
        s = time.time()

    if args.detector_name in ["Florence-2-large-v2", "Florence-2-large-ft-v2"]:
        model, processor = load_florence_2(detector_name=args.detector_name, main_folder=args.main_folder)
    else:
        raise Exception(f"Not implemented for {args.detector_name}")

    save_prepro_step_2_folder = os.path.join(
        args.main_folder,
        args.dataset,
        args.detector_name,
        "new_outputs_prepro_step_2",
        args.split,
    )
    if not os.path.exists(save_prepro_step_2_folder):
        os.makedirs(save_prepro_step_2_folder, exist_ok=True)
        print("mkdir", save_prepro_step_2_folder)

    c_missing_gdino_inf, c_nb_convert_rgb = 0, 0

    for it, (image_id, image, whwh, sent_id, sent, GT_bbox) in tqdm(enumerate(data)):
        if it % 10 == 0:
            print(f"it {it}")

        filename = f"IMG_ID_{image_id}_SENT_ID_{sent_id}.pt"  # nomenclature for full sent results

        if not os.path.exists(os.path.join(save_prepro_step_2_folder, filename)):
            c_missing_gdino_inf += 1

            with torch.no_grad():

                multi_task_results = {}

                # 1) OVD / REC (use in LLM-wrapper + for VLM baseline)
                # text input, output boxes + labels
                try:
                    results = run_example(
                        model=model,
                        processor=processor,
                        image=image,
                        task_prompt="<OPEN_VOCABULARY_DETECTION>",
                        text_input=sent,
                    )
                    results = convert_to_od_format(results["<OPEN_VOCABULARY_DETECTION>"])
                    results["original_text_query"] = sent
                    multi_task_results["ORI_QUERY_REC_like"] = results
                except:
                    c_nb_convert_rgb += 1
                    results = run_example(
                        model=model,
                        processor=processor,
                        image=image.convert("RGB"),
                        task_prompt="<OPEN_VOCABULARY_DETECTION>",
                        text_input=sent,
                    )
                    results = convert_to_od_format(results["<OPEN_VOCABULARY_DETECTION>"])
                    results["original_text_query"] = sent
                    multi_task_results["ORI_QUERY_REC_like"] = results

                # 2) original query grounding
                # text input, output boxes + labels <- ground mentioned objects in the scene
                try:
                    results = run_example(
                        model=model,
                        processor=processor,
                        image=image,
                        task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
                        text_input=sent,
                    )
                    results["<CAPTION_TO_PHRASE_GROUNDING>"]["original_text_query"] = sent
                    multi_task_results["ORI_QUERY_GROUNDING"] = results["<CAPTION_TO_PHRASE_GROUNDING>"]
                except:
                    results = run_example(
                        model=model,
                        processor=processor,
                        image=image.convert("RGB"),
                        task_prompt="<CAPTION_TO_PHRASE_GROUNDING>",
                        text_input=sent,
                    )
                    results["<CAPTION_TO_PHRASE_GROUNDING>"]["original_text_query"] = sent
                    multi_task_results["ORI_QUERY_GROUNDING"] = results["<CAPTION_TO_PHRASE_GROUNDING>"]

                # 3) region captioning (optional, can boost LLM-wrapper scores)
                # 0 input, output boxes + labels <- ground found objects in the scene
                try:
                    results = run_example(
                        model=model,
                        processor=processor,
                        image=image,
                        task_prompt="<DENSE_REGION_CAPTION>",
                    )
                    multi_task_results["DENSE_REGION_CAPTION"] = results["<DENSE_REGION_CAPTION>"]
                except:
                    results = run_example(
                        model=model,
                        processor=processor,
                        image=image.convert("RGB"),
                        task_prompt="<DENSE_REGION_CAPTION>",
                    )
                    multi_task_results["DENSE_REGION_CAPTION"] = results["<DENSE_REGION_CAPTION>"]

                # Print & save info
                if verbose:
                    if it < 15:
                        # print inferred data as text
                        print(f"\n\n It {it}:")
                        for key in multi_task_results.keys():
                            print(f"\nkey {key}")
                            for sub_key in multi_task_results[key]:
                                print(f"{sub_key}")
                                print(f"{multi_task_results[key][sub_key]}")

                    # save vizu per task prompt
                    if it < 50:
                        if not os.path.exists(os.path.join(save_prepro_step_2_folder, "vizu", filename.strip(".pt"))):
                            os.makedirs(
                                os.path.join(
                                    save_prepro_step_2_folder,
                                    "vizu",
                                    filename.strip(".pt"),
                                ),
                                exist_ok=True,
                            )

                        for key in multi_task_results.keys():
                            plot_bbox(
                                image=image,
                                box_data=multi_task_results[key],
                                vizu_path=os.path.join(
                                    save_prepro_step_2_folder,
                                    "vizu",
                                    filename.strip(".pt"),
                                    key + ".png",
                                ),
                            )

                torch.save(
                    multi_task_results,
                    os.path.join(save_prepro_step_2_folder, filename),
                )

    print("\nTotal number of data not already preprocessed:", c_missing_gdino_inf)
    print("Total number of data needing convert_rgb:", c_nb_convert_rgb)

    if timing:
        e = time.time()
        print("Florence 2 preprocessing time:", round(e - s, 1), "secs")

    return None


def postpro_step_2_florence2(args, loader, save_postpro_filename, timing=True, verbose=False):

    if timing:
        step_2_s = time.time()

    if args.detector_name not in ["Florence-2-large-v2", "Florence-2-large-ft-v2"]:
        raise Exception(f"Not implemented for {args.detector_name}")

    print("\nStart postprocessing for Florence-2")

    save_prepro_step_2_folder = os.path.join(
        args.main_folder,
        args.dataset,
        args.detector_name,
        "new_outputs_prepro_step_2",
        args.split,
    )

    all_info_step_2 = []
    issues_recap = {
        "ORI_QUERY_REC_like": {
            "c_issue_too_many": 0,
            "c_issue_coord_ord": 0,
            "c_both_issues": 0,
            "how_many_too_many_boxes": [],
            "how_many_unclean_boxes": [],
            "how_many_clean_boxes": [],
        },
        "ORI_QUERY_GROUNDING": {
            "c_issue_too_many": 0,
            "c_issue_coord_ord": 0,
            "c_both_issues": 0,
            "how_many_too_many_boxes": [],
            "how_many_unclean_boxes": [],
            "how_many_clean_boxes": [],
        },
        "DENSE_REGION_CAPTION": {
            "c_issue_too_many": 0,
            "c_issue_coord_ord": 0,
            "c_both_issues": 0,
            "how_many_too_many_boxes": [],
            "how_many_unclean_boxes": [],
            "how_many_clean_boxes": [],
        },
    }

    for it, (image_id, whwh, sent_id, sent, GT_bbox) in tqdm(enumerate(loader)):

        # 1) Init info_step_2 dict with GT data

        image_id_or_name = image_id.item() if "RefCOCO" in args.dataset else image_id[0]

        info_step_2 = {
            "image_id": image_id_or_name,
            "image_width": whwh[0, 0].item(),
            "image_height": whwh[0, 1].item(),
            "category_id": sent_id.item(),
            "original_full_query": sent[0],
        }

        GT_bbox = box_convert(
            GT_bbox, in_fmt="xywh", out_fmt="xyxy"
        )  # we need this xyxy format during REC eval step (box_iou)
        info_step_2["GT_bbox"] = GT_bbox.detach().cpu()[0]  # format xyxy, original image scale

        # 2) Process full sent results

        filename = f"IMG_ID_{image_id_or_name}_SENT_ID_{sent_id.item()}.pt"
        if not os.path.exists(os.path.join(save_prepro_step_2_folder, filename)):
            raise Exception(f"Full sentence not preprocessed for img id {image_id_or_name} and sent id {sent_id.item()}")

        results = torch.load(os.path.join(save_prepro_step_2_folder, filename))

        with torch.no_grad():

            # All results under this format
            #  [{'query': 'a bush',
            #    'nb_box_found': 2,
            #    'found_boxes': tensor([[352.9600,  87.7485, 149.7600, 125.5380],
            #            [ 80.0000,  95.0075, 129.9200, 127.2460]])},
            #   ...
            #   {'query': 'woman',
            #    'nb_box_found': 1,
            #    'found_boxes': tensor([[346.2400, 260.6835, 146.5600, 258.7620]])}]}

            # 2.1) Add REC like data

            rec_box_exist = False
            if "ORI_QUERY_REC_like" in results:
                REC_bbox_xyxy = results["ORI_QUERY_REC_like"][
                    "bboxes"
                ]  # format xyxy, original image scale (1 list of X sublists of len 4)
                if REC_bbox_xyxy != [[]] and REC_bbox_xyxy != []:
                    rec_box_exist = True
                    REC_bbox_cxcywh = (
                        box_convert(
                            torch.tensor(REC_bbox_xyxy[0], device="cpu").unsqueeze(0),
                            in_fmt="xyxy",
                            out_fmt="cxcywh",
                        )
                        .squeeze(0)
                        .cpu()
                    )
                    info_step_2["REC"] = {
                        # format cxcywh, original image scale
                        # (one array of size 4, except in rare cases where more than 1 rec box found)
                        "box": REC_bbox_cxcywh,
                        "label": results["ORI_QUERY_REC_like"]["labels"][0],
                    }
            if not rec_box_exist:
                print("\nNo rec box found for it", it)
                info_step_2["REC"] = {"box": [], "label": []}

            # 2.2) Add all other type of boxes

            for key in results.keys():
                list_of_boxes_per_label = []
                bboxes_xyxy = results[key]["bboxes"]

                zero_box_detected = True
                if bboxes_xyxy != [[]] and bboxes_xyxy != []:
                    zero_box_detected = False
                    issues_recap[key]["how_many_unclean_boxes"].append(len(bboxes_xyxy))
                    bboxes_xyxy = torch.tensor(bboxes_xyxy, device="cpu")
                    issue_with_coord_ord = False

                    if (
                        torch.sum(bboxes_xyxy[:, 0] > bboxes_xyxy[:, 2]).item() > 0
                        or torch.sum(bboxes_xyxy[:, 1] > bboxes_xyxy[:, 3]).item() > 0
                    ):
                        issue_with_coord_ord = True
                        unclean_boxes_nb = len(bboxes_xyxy)
                        # erasing rows with coordinates that do not verify proper xyxy format
                        compar_1 = bboxes_xyxy[:, 0] < bboxes_xyxy[:, 2]
                        compar_2 = bboxes_xyxy[:, 1] < bboxes_xyxy[:, 3]
                        good_rows_mask = torch.min(compar_1, compar_2)
                        bboxes_xyxy = bboxes_xyxy[good_rows_mask, :]
                        if len(bboxes_xyxy) == 0:
                            zero_box_detected = True
                        if verbose:
                            print(f"it for key {key}")
                            print(f"{unclean_boxes_nb-len(bboxes_xyxy)} issues with coordinates ordering")
                        issues_recap[key]["c_issue_coord_ord"] = issues_recap[key]["c_issue_coord_ord"] + (
                            unclean_boxes_nb - len(bboxes_xyxy)
                        )

                    issues_recap[key]["how_many_clean_boxes"].append(len(bboxes_xyxy))

                    if len(bboxes_xyxy) > 30:
                        if not issue_with_coord_ord:
                            if verbose:
                                print(f"it for key {key}")
                        else:
                            issues_recap[key]["c_both_issues"] = issues_recap[key]["c_both_issues"] + 1
                        if verbose:
                            print(
                                "case with more than 30 boxes:",
                                len(bboxes_xyxy),
                                "boxes",
                            )
                        issues_recap[key]["c_issue_too_many"] = issues_recap[key]["c_issue_too_many"] + 1
                        issues_recap[key]["how_many_too_many_boxes"].append(len(bboxes_xyxy))

                if not zero_box_detected:
                    bboxes_cxcywh = box_convert(
                        bboxes_xyxy, in_fmt="xyxy", out_fmt="cxcywh"
                    )  # dim total_nb_thres_box x 4, cxcywh format, ori img size
                    labels = [e.lower().strip() for e in results[key]["labels"]]
                    labels_counter = Counter(labels)

                    list_of_boxes_per_label = [
                        {
                            "query": one_label,
                            "nb_box_found": labels_counter[one_label],
                            "found_boxes": torch.zeros((0, 4), device="cpu"),
                        }
                        for one_label in labels_counter
                    ]
                    for one_box, one_label in zip(bboxes_cxcywh, labels):
                        for query_dict in list_of_boxes_per_label:
                            if one_label == query_dict["query"]:
                                query_dict["found_boxes"] = torch.cat(
                                    (query_dict["found_boxes"], one_box.unsqueeze(0)),
                                    dim=0,
                                )

                else:
                    print("\nwarning: no boxes found for it", it, "and key", key)
                    list_of_boxes_per_label = [{"query": "None", "nb_box_found": 0, "found_boxes": []}]
                    issues_recap[key]["how_many_unclean_boxes"].append(0)
                    issues_recap[key]["how_many_clean_boxes"].append(0)

                info_step_2[key] = list_of_boxes_per_label

        # 3) Append global list
        all_info_step_2.append(info_step_2)

    for key in issues_recap.keys():
        print(f"\n\nFor key {key}")
        print(f"c_issue_too_many={issues_recap[key]['c_issue_too_many']}")
        print(f"c_issue_coord_ord={issues_recap[key]['c_issue_coord_ord']}")
        print(f"c_both_issues={issues_recap[key]['c_both_issues']}")
        if len(issues_recap[key]["how_many_too_many_boxes"]) > 0:
            print(
                "avg how_many_too_many_boxes:",
                sum(issues_recap[key]["how_many_too_many_boxes"]) / len(issues_recap[key]["how_many_too_many_boxes"]),
            )
            print(
                "max nb of too_many_boxes:",
                max(issues_recap[key]["how_many_too_many_boxes"]),
            )

        print(
            "\nAvg how_many_clean_boxes:",
            sum(issues_recap[key]["how_many_clean_boxes"]) / len(issues_recap[key]["how_many_clean_boxes"]),
        )
        print(
            "\nAvg how_many_unclean_boxes:",
            sum(issues_recap[key]["how_many_unclean_boxes"]) / len(issues_recap[key]["how_many_unclean_boxes"]),
        )

    if not args.dont_save_vlm_postpro:
        save_path = os.path.join(
            args.main_folder,
            args.dataset,
            args.detector_name,
            "new_outputs_postpro_step_2",
        )

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        print("Saving results to file")
        torch.save(all_info_step_2, save_postpro_filename)

    if timing:
        step_2_e = time.time()
        print("Step 2 postpro time:", round(step_2_e - step_2_s, 3), "secs")

    return all_info_step_2
