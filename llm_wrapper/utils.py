import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def convert_box_coord_list(one_box_list, out_format, coord_noise=False):

    # input format is cxcywh
    if out_format == "cxcywh":
        return one_box_list

    # coord conversion
    cx, cy, w, h = one_box_list
    x_min = cx - (w / 2)
    y_min = cy - (h / 2)
    if out_format == "xywh":
        result = [x_min, y_min, w, h]
    elif out_format == "xyxy":
        x_max = x_min + w
        y_max = y_min + h
        result = [x_min, y_min, x_max, y_max]

    # add noise / round each coord if needed
    if coord_noise:
        noise = np.random.uniform(-5, 5, 4)
        result = list(noise + np.array(result))

    # correction of the conversion / noise cases that give results such as -0.0 or -0.1
    if result[0] <= 0:
        result[0] = 0.0
    if result[1] <= 0:
        result[1] = 0.0

    return result


def print_all_rec_in_folder(folder_path):

    filename = os.path.join(folder_path, "recap.txt")
    outfile = open(filename, "w+")

    all_json_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and "checkpoint" in f]
    file_step_nb = [int(elem.replace("checkpoint-", "").replace(".json", "")) for elem in all_json_files]
    file_step_nb.sort()
    reordered_json_files = ["checkpoint-" + str(e) + ".json" for e in file_step_nb]

    best_rec = 0
    for file in reordered_json_files:
        with open(os.path.join(folder_path, file)) as f:
            data = json.load(f)
        ckpt_nb = file.replace("checkpoint-", "").replace(".json", "")
        if round(data["REC_metric"] * 100, 2) > best_rec:
            best_rec = round(data["REC_metric"] * 100, 2)
            best_ckpt_nb = ckpt_nb
        print(
            f"ckpt {ckpt_nb}: REC = {round(data['REC_metric']*100,2)} on {data['nb_LLM_infer']} "
            f"(iou = {round(data['avg_iou']*100,2)})"
        )
        outfile.write(
            f"\nckpt {ckpt_nb}: REC = {round(data['REC_metric']*100,2)} on {data['nb_LLM_infer']} "
            f"(iou = {round(data['avg_iou']*100,2)})"
        )

    print(f"\nBest REC = {best_rec} for ckpt {best_ckpt_nb}")
    outfile.write(f"\n\nBest REC = {best_rec} for ckpt {best_ckpt_nb}")
    outfile.close()
    return None


def show_box(box, ax, box_format="xywh", ori_width=1, ori_height=1, box_label="", col="red"):
    x0, y0 = box[0] * ori_width, box[1] * ori_height
    if box_format == "xyxy":  # box format: [xmin, ymin, xmax, ymax]
        w, h = (box[2] - box[0]) * ori_width, (box[3] - box[1]) * ori_height
    elif box_format == "xywh":  # box format: [xmin, ymin, w, h]
        w, h = box[2] * ori_width, box[3] * ori_height
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=col, facecolor=(0, 0, 0, 0), lw=3, label=box_label))
    return None


def visu_refcocog(
    refer,
    main_folder,
    image_id,
    text_no_modif=None,
    text_modif=None,
    box_no_modif=None,
    box_modif=None,
    box_GT=None,
    box_GT_format="xyxy",
    title_text=None,
    save_filepath=None,
):

    # load image
    IMAGE_DIR = os.path.join(main_folder, "COCO", "images", "train2014")
    image_info = refer.loadImgs(image_ids=[image_id])[0]
    img_path = os.path.join(IMAGE_DIR, image_info["file_name"])
    img = np.asarray(Image.open(img_path))

    # show
    fig = plt.figure()
    plt.imshow(img)
    if box_no_modif is not None:
        show_box(
            box=box_no_modif,
            ax=plt.gca(),
            box_format="xyxy",
            box_label=text_no_modif,
            col="blue",
        )
    if box_modif is not None:
        show_box(
            box=box_modif,
            ax=plt.gca(),
            box_format="xyxy",
            box_label=text_modif,
            col="orange",
        )
    if box_GT is not None:
        show_box(
            box=box_GT,
            ax=plt.gca(),
            box_format=box_GT_format,
            box_label="GT box",
            col="green",
        )
    plt.axis("off")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if title_text is not None:
        plt.title(title_text)
    if save_filepath is not None:
        plt.savefig(save_filepath, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return None
