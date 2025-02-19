import argparse


def boolean_flag(arg) -> bool:
    """Add a boolean flag to argparse parser."""
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ("true", "1", "yes", "y"):
        return True
    elif arg.lower() in ("false", "0", "no", "n"):
        return False
    else:
        raise ValueError(f"Expected 'true'/'false' or '1'/'0', but got '{arg}'")


def None_or_int(arg):

    if arg is None:
        return None
    elif arg.lower() in ["none"]:
        return None
    else:
        return int(arg)


def get_args(notebook=False):

    parser = argparse.ArgumentParser(
        description="Args to finetune or to infer with LLM-wrapper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------------------
    # Path that a user would need to change
    # ---------------------------------------------------

    parser.add_argument(
        "--HF_token_path",
        type=str,
        default="./llm_wrapper/LLM_token.json",
        help="Path to HuggingFace's token with proper rights to access chosen LLM (in working directory by default)",
    )

    parser.add_argument(
        "--main_folder",
        type=str,
        default="./llm_wrapper/data",
        help="Path to main folder where to save/load files (datasets, base LLMs, fine-tuned LLM checkpoints)",
    )

    # ---------------------------------------------------
    # Args specific to VLM_inference_script.py
    # ---------------------------------------------------

    parser.add_argument(
        "--not_vlm_prepro",
        action="store_true",
        default=False,
        help="Whether to do VLM preprocessing (basic VLM inference)",
    )

    parser.add_argument(
        "--not_vlm_postpro",
        action="store_true",
        default=False,
        help="Whether to do VLM postprocessing (thresholding, formatting for LLM-wrapper pipeline, etc)",
    )

    parser.add_argument(
        "--dont_save_vlm_postpro",
        action="store_true",
        default=False,
        help="Whether to save VLM postpro results",
    )

    # specific to GDrec
    parser.add_argument(
        "--tok_logit_operation",
        "--log_op",
        type=str,
        default="min",
        choices=["max", "min", "avg"],
        help=(
            "What reduction operation is applied on last dimension of the logit submatrix of size "
            "900xnb_tokens_in_current_noun_group (for GDrec)"
        ),
    )

    # specific to GDrec
    parser.add_argument(
        "--step_1_method",
        type=str,
        default="spacy_trf",
        choices=["spacy_sm", "spacy_trf"],
        help="Method used to extract short queries during GDrec text preprocessing",
    )

    # specific to GDrec
    parser.add_argument(
        "--dont_get_all_extract_scores_wo_pooling",
        action="store_true",
        default=False,
        help="Whether to extract from GDrec inference an unpooled score matrix of: noun group x 900 box x max nb token",
    )

    # specific to Florence 2
    parser.add_argument(
        "--spe_fall_back_for_vlm_0s",
        action="store_true",
        default=False,
        help=(
            "Whether to use fallback strategy when using 0-shot Florence-2 "
            "(if no REC box found, use the first grounding box)"
        ),
    )

    # ---------------------------------------------------
    # Args specific to LLM_finetuning_scrpt.py
    # ---------------------------------------------------

    parser.add_argument("--epoch", type=int, default=1, help="Number of fine-tuning epochs")

    parser.add_argument("--BS", type=int, default=4, help="Fine-tuning batch size")

    parser.add_argument("--LR", type=float, default=1e-5, help="Fine-tuning learning rate")

    parser.add_argument(
        "--LORA_r",
        "--r",
        type=int,
        default=128,
        help="Rank param (r) when fine-tuning with LoRA",
    )

    parser.add_argument(
        "--LORA_alpha",
        "--a",
        type=int,
        default=256,
        help="Alpha param when fine-tuning with LoRA",
    )

    parser.add_argument(
        "--modules_to_FT",
        "--modules",
        type=str,
        default=None,
        choices=[None, "last_4_only", "lm_head_only"],
        help="Choice of the modules in the LLM to fine-tune with LoRA",
    )

    parser.add_argument(
        "--ckpts_save_strategy",
        type=str,
        default="steps",
        choices=["steps", "no", "epoch"],
        help="Specify checkpoint saving strategy when fine-tuning",
    )

    parser.add_argument(
        "--ckpt_save_steps",
        type=int,
        default=500,
        help="We save LLM checkpoints at every step that is a multiple of this value",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        "--resume",
        action="store_true",
        default=False,
        help="Whether to resume fine-tuning from last existing checkpoint (if True) or from scratch (if False)",
    )

    parser.add_argument(
        "--max_len_train",
        type=int,
        default=None,
        help="Number of fine-tuning data to use (if None then all the fine-tuning data in specified file is used)",
    )

    parser.add_argument(
        "--max_len_val",
        type=int,
        default=500,
        help="Number of val data to use during fine-tuning",
    )

    parser.add_argument(
        "--eval_on_train_batch",
        action="store_true",
        default=False,
        help="Whether to eval on a train batch (if True) instead of a val split during fine-tuning",
    )

    parser.add_argument(
        "--Nb_train_data",
        type=int,
        default=None,
        help="To be specified to find the right file containing the proper fine-tuning train data",
    )

    parser.add_argument(
        "--Nb_val_data",
        type=int,
        default=2000,
        help="To be specified to find the right file containing the proper fine-tuning val data",
    )

    parser.add_argument(
        "--use_callbacks",
        action="store_true",
        default=False,
        help="Whether to use callbacks during fine-tuning to see a few generated data (slows down the fine-tuning)",
    )

    parser.add_argument(
        "--max_eval_data",
        type=int,
        default=30,
        help="Number max of fine-tuning / val data to use for eval in PrintCallback (if --use_callbacks)",
    )

    parser.add_argument(
        "--nb_good_idx",
        type=int,
        default=5,
        help="Number of data that should be found zero shot amongst max_eval_data in our_eval (if --use_callbacks)",
    )

    # ---------------------------------------------------------------------------------
    # Args specific to the creation of fine-tuning dataset (LLM_inference_script.py)
    # ---------------------------------------------------------------------------------

    parser.add_argument(
        "--create_FT_dataset",
        "--FT",
        action="store_true",
        default=False,
        help="Whether to create a FT dataset (if True) instead of doing classic inference with LLM-wrapper (if False)",
    )

    parser.add_argument(
        "--viz_some_FT_data",
        action="store_true",
        default=False,
        help=(
            "Whether to visualize and save some fine-tuning data while creating it for RefCOCO/+/g "
            "-> requires REFER installed"
        ),
    )

    # -------------------------------------------------------------------------------
    # Args specific to VLM/LLM-wrapper inference and REC eval (LLM_inference_script.py)
    # -------------------------------------------------------------------------------

    parser.add_argument(
        "--limit_to_300_samples_GDINO_1_5",
        action="store_true",
        default=False,
        help="Whether to limit RefCOCOg val eval to the same 300 data used for our GDINO 1.5 eval",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train", "testA", "testB"],
        help="Specify which split to use for chosen dataset",
    )

    parser.add_argument(
        "--min_samples",
        "--min_s",
        type=int,
        default=0,
        help="Min sample index to use during inference",
    )

    parser.add_argument(
        "--max_samples",
        "--max_s",
        "--s",
        type=int,
        default=500,
        help="Max sample index to use during inference",
    )

    parser.add_argument(
        "--max_gen_tok",
        type=int,
        default=128,
        help="Max number of newly generated tokens during LLM generation",
    )

    parser.add_argument(
        "--VLM_only",
        action="store_true",
        default=False,
        help="Whether to use VLM only for REC evaluation (and avoid applying LLM-wrapper)",
    )

    # below: args only for our large scale eval of checkpoints,
    # all requiring specifying --folder_with_checkpoints_to_eval
    parser.add_argument(
        "--folder_with_checkpoints_to_eval",
        type=str,
        default=None,
        help=(
            "Path to folder where script will load and eval all checkpoints if not None "
            "(if set to 'auto', the folder name is automatically found given the other arguments)"
        ),
    )

    parser.add_argument(
        "--min_checkpt",
        type=int,
        default=500,
        help="Min step value of checkpoints to eval if specify --folder_with_checkpoints_to_eval",
    )

    parser.add_argument(
        "--max_checkpt",
        type=int,
        default=100000,
        help="Max step value of checkpoints to eval if specify --folder_with_checkpoints_to_eval",
    )

    parser.add_argument(
        "--ckpt_multiple",
        "--ckpt_mult",
        type=int,
        default=None,
        help=(
            "Eval only checkpoints whose step values are a multiple of this value "
            "(if specify --folder_with_checkpoints_to_eval)"
        ),
    )

    parser.add_argument(
        "--last_checkpt",
        action="store_true",
        default=False,
        help="Eval only the last checkpoint in given --folder_with_checkpoints_to_eval",
    )

    parser.add_argument(
        "--exact_checkpt_list",
        type=int,
        nargs="+",
        default=None,
        help="Exact list of checkpoints step numbers to eval if specify --folder_with_checkpoints_to_eval",
    )

    # ----------------------------------------------------------------------------------------------
    # Args for both inference (LLM_inference_script.py) and fine-tuning (LLM_finetuning_script.py)
    # ----------------------------------------------------------------------------------------------

    parser.add_argument(
        "--detector_name",
        "--detector",
        "--VLM",
        type=str,
        default="Florence-2-large-v2",
        choices=[
            "GDINO_SwinT_original",
            "GDINO_SwinB_original",
            "GDINO_SwinT",
            "GDINO_SwinB",
            "GDINO_1.5",
            "Florence-2-large-v2",
            "Florence-2-large-ft-v2",
            "KOSMOS",
        ],
        help="VLM (for open voc object detection) to use or adapt with LLM-wrapper",
    )

    parser.add_argument(
        "--detector_name_bis",
        "--detector_bis",
        type=str,
        default=None,
        choices=[
            "GDINO_SwinT_original",
            "GDINO_SwinB_original",
            "GDINO_SwinT",
            "GDINO_SwinB",
            "GDINO_1.5",
            "Florence-2-large-v2",
            "Florence-2-large-ft-v2",
            "KOSMOS",
        ],
        help=(
            "Second VLM (for open voc object detection) to use or adapt with LLM-wrapper "
            "when doing ensembling (if set to None, we only use --detector_name)"
        ),
    )

    parser.add_argument(
        "--HF_LLM_name",
        "--n",
        "--LLM",
        type=str,
        default="Meta-Llama-3-8B-Instruct",  # "Mixtral-8x7B-Instruct-v0.1",
        help=(
            "LLM to use (for inference or fine-tuning): "
            "either base HuggingFace model name or name of checkpoint folder if use finetuned LLM"
        ),
    )

    parser.add_argument(
        "--prompt_version",
        "--p",
        type=int,
        default=31,  # in paper, 32 when using GD/GDrec, 31 for other VLMs (Florence 2, etc)
        choices=[0, 31, 32],
        help=(
            "Prompt template to use for LLM-wrapper (0 = Oracle (avoid prompt and best candidate is automatically selected) "
            "VS 31 = prompt with coordinates VS 32 = prompt with coordinates and relevance scores)"
        ),
    )

    parser.add_argument(
        "--prompt_version_2",
        "--p_2",
        type=int,
        default=32,
        help="Same as --prompt_version but for detector_name_bis if we use it (if detector_name_bis is not None)",
    )

    parser.add_argument(
        "--prompt_context",
        type=int,
        default=18,  # in paper, 0 for GD, 4 for GDrec, 18 for Florence-2
        choices=[0, 4, 18, 20],
        help=(
            "Nature of boxes that we include in the prompt (0 only includes grounding boxes (for GD/GDrec), "
            "4 resp. 18 includes REC and grounding boxes (for GD/GDrec resp. Florence-2), "
            "20 includes REC, grounding and dense region caption (for Florence-2)"
        ),
    )

    parser.add_argument(
        "--prompt_context_2",
        type=int,
        default=0,
        choices=[0, 4, 18, 20],
        help="Same as --prompt_context but for detector_name_bis if we use it (if detector_name_bis is not None)",
    )

    parser.add_argument(
        "--max_nb_of_box_proposal",
        type=None_or_int,
        default=None,
        help="Maximum number of box proposal used to build the prompt (if None, we keep them all)",
    )

    parser.add_argument(
        "--bbox_format",
        "--bbox",
        type=str,
        default="xyxy",
        choices=["xyxy", "xywh", "cxcywh"],
        help="Coordinate format to use in the prompt",
    )

    parser.add_argument(
        "--box_t_subquery",
        type=float,
        default=0.2,
        help="Box relevance threshold to use for GDrec (for all queries besides the subject and the full sentence)",
    )

    parser.add_argument(
        "--box_t_subject",
        type=float,
        default=0.15,
        help="Box relevance threshold to use for GDrec for all subject queries",
    )

    parser.add_argument(
        "--box_t_full_sent",
        type=float,
        default=0.2,  # in paper, 0.2 for GD & 0.3 for GDrec
        help=(
            "Box relevance threshold to use for GD / GDrec for full sentence queries "
            "(in particular, 0.3 for GDrec & 0.2 for GD)"
        ),
    )

    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.05,
        help="Text threshold to use for GD",
    )

    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers to use")

    parser.add_argument(
        "--dataset",
        type=str,
        default="RefCOCOg_umd",
        choices=[
            "RefCOCOg_umd",
            "RefCOCO_unc",
            "RefCOCO+_unc",
            "talk2car",
            "HC_RefLoCo",
        ],
        help="Dataset to use for eval (same as the one used to fine-tune the LLM if ft_dataset is None)",
    )

    parser.add_argument(
        "--ft_dataset",
        type=str,
        default=None,
        choices=[
            "RefCOCOg_umd",
            "RefCOCO_unc",
            "RefCOCO+_unc",
            "talk2car",
            "HC_RefLoCo",
        ],
        help="Dataset that was used to fine-tune the LLM checkpoint we want to load (if different from --dataset)",
    )

    parser.add_argument(
        "--FT_data_modif",
        type=str,
        default="aug",
        choices=["aug", "shuffle", "no_modif"],
        help="Modif to apply to the fine-tuning data (whether to augment or shuffle the data) if we create or use it",
    )

    parser.add_argument(
        "--dont_save_results",
        action="store_true",
        default=False,
        help="Whether to save (in json format) REC/IOU metrics + a list of dataset indices correctly inferred by the model",
    )

    parser.add_argument(
        "--verbose",
        "--v",
        action="store_true",
        default=False,
        help="Whether to activate verbose to print additional information",
    )

    if notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    return args
