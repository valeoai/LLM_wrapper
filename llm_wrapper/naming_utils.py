import os


def get_VLM_preprocessed_files_paths(args):

    # Sanity check
    if args.limit_to_300_samples_GDINO_1_5:
        if args.dataset != "RefCOCOg_umd" or args.split != "val" or args.max_samples != 4896:
            raise Exception(
                "Please use split = val and max_samples = 4896 to properly eval with limit_to_300_samples_GDINO_1_5"
            )

    if args.min_samples >= args.max_samples:
        raise Exception(f"Sorry, we must have min_sample idx({args.min_samples}) < max_sample idx ({args.max_samples})")

    # Print important args
    print(f"\n\nEval on samples [{args.min_samples} to {args.max_samples - 1}] from {args.dataset} {args.split}")
    if args.VLM_only:
        print(f"{args.detector_name} only")
    else:
        print(f"\nVLM preprocessing: {args.detector_name}")
        if args.detector_name not in ["GDINO_SwinT", "GDINO_SwinB"]:
            print(f"(box_t_full_sent = {args.box_t_full_sent})")
        elif args.detector_name in ["GDINO_SwinT", "GDINO_SwinB"]:
            print(f"Text preprocessing: {args.step_1_method}")
            print(
                f"logit operation = {args.tok_logit_operation} | (box_t_subject = {args.box_t_subject} "
                f"| box_t_subquery = {args.box_t_subquery} | box_t_full_sent = {args.box_t_full_sent})"
            )
        short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=args.HF_LLM_name)
        print(
            f"LLM inference: {short_LLM_name} | Prompt {args.prompt_version} "
            f"| context end {args.prompt_context} | {args.bbox_format} coords"
        )

    # Create folder for FT data if needed
    if args.create_FT_dataset:
        if args.FT_data_modif == "aug":
            print("build ft dataset with augmentation")
        elif args.FT_data_modif == "shuffle":
            print("build ft dataset with shuffle_box_order")
        else:
            print("build ft dataset without augmentation nor shuffle_box_order")

    # Get dataset if needed
    if args.dataset == "talk2car":
        data_name = "talk2car"
        if args.split == "val":
            full_max_sample = 1163
        elif args.split == "test":
            full_max_sample = 2447
        elif args.split == "train":
            full_max_sample = 8348

    if args.dataset == "HC_RefLoCo":
        data_name = "HC_RefLoCo"
        assert args.split != "train"
        if args.split == "val":
            full_max_sample = 13360
        elif args.split == "test":
            full_max_sample = 31378

    if args.dataset == "RefCOCOg_umd":
        data_name = "Refcocog"
        if args.split == "val":
            full_max_sample = 4896
        elif args.split == "test":
            full_max_sample = 9602
        elif args.split == "train":
            full_max_sample = 80512

    elif args.dataset == "RefCOCO_unc":
        data_name = "Refcoco"
        if args.split == "val":
            full_max_sample = 10834
        elif args.split == "test":
            full_max_sample = 10752
        elif args.split == "train":
            full_max_sample = 120624
        elif args.split == "testA":
            full_max_sample = 5657
        elif args.split == "testB":
            full_max_sample = 5095

    elif args.dataset == "RefCOCO+_unc":
        data_name = "Refcoco+"
        if args.split == "val":
            full_max_sample = 10758
        elif args.split == "test":
            full_max_sample = 10615
        elif args.split == "train":
            full_max_sample = 120191
        elif args.split == "testA":
            full_max_sample = 5726
        elif args.split == "testB":
            full_max_sample = 4889

    split_str = "_" + args.split
    length = args.max_samples
    articles = "_wo_articles"

    # Get paths for text preprocessing if needed
    if args.detector_name in [
        "GDINO_SwinT",
        "GDINO_SwinB",
        "GDINO_SwinT_original",
        "GDINO_SwinB_original",
    ]:
        step_1_folder = (
            "with_noun_group_strings" if args.detector_name in ["GDINO_SwinT", "GDINO_SwinB"] else "with_all_token_strings"
        )
        step_1_path = os.path.join(args.main_folder, args.dataset, "new_outputs_step_1", step_1_folder)
        if not os.path.exists(step_1_path):
            os.makedirs(step_1_path, exist_ok=True)
        filepath_step_1 = os.path.join(
            step_1_path,
            f"Result_step_1_{data_name}{split_str}_{args.min_samples}_{length}_samples_with_{args.step_1_method}"
            + articles
            + ".json",
        )
        full_filepath_step_1 = os.path.join(
            step_1_path,
            f"Result_step_1_{data_name}_{args.split}_0_{full_max_sample}_samples_with_{args.step_1_method}"
            + articles
            + ".json",
        )
        if not os.path.exists(full_filepath_step_1):
            print("\nfull_filepath_step_1 does not exist at:", full_filepath_step_1)
    else:
        filepath_step_1, full_filepath_step_1 = None, None

    # Get paths for VLM (text-image) pairs preprocessing for VLM 1
    log_op_str = f"_log_op_{args.tok_logit_operation}" if (args.detector_name in ["GDINO_SwinB", "GDINO_SwinT"]) else ""
    if args.detector_name not in ["Florence-2-large-v2", "Florence-2-large-ft-v2"]:
        filename_step_2 = (
            f"{data_name}_{args.split}_{args.min_samples}_{length}_samples_with_{args.step_1_method}"
            + articles
            + (
                f"{log_op_str}_box_t_subject_{args.box_t_subject}_box_t_subquery_{args.box_t_subquery}"
                f"_box_t_full_sent_{args.box_t_full_sent}_cxcywh_format.pt"
            )
        )
        full_filename_step_2 = (
            f"{data_name}_{args.split}_0_{full_max_sample}_samples_with_{args.step_1_method}"
            + articles
            + (
                f"{log_op_str}_box_t_subject_{args.box_t_subject}_box_t_subquery_{args.box_t_subquery}"
                "_box_t_full_sent_{args.box_t_full_sent}_cxcywh_format.pt"
            )
        )
    elif args.detector_name in ["Florence-2-large-v2", "Florence-2-large-ft-v2"]:
        filename_step_2 = f"{data_name}_{args.split}_{args.min_samples}_{length}_samples_cxcywh_format.pt"
        full_filename_step_2 = f"{data_name}_{args.split}_0_{full_max_sample}_samples_cxcywh_format.pt"
    filepath_step_2 = os.path.join(
        args.main_folder,
        args.dataset,
        args.detector_name,
        "new_outputs_postpro_step_2",
        filename_step_2,
    )
    full_filepath_step_2 = os.path.join(
        args.main_folder,
        args.dataset,
        args.detector_name,
        "new_outputs_postpro_step_2",
        full_filename_step_2,
    )

    if not os.path.exists(full_filepath_step_2):
        print("\nfull_filepath_step_2 does not exist at:", full_filepath_step_2)

    # Same for VLM 2 if specified
    if args.detector_name_bis is None:
        filepath_step_2_bis, full_filepath_step_2_bis = None, None
    else:
        log_op_str = (
            f"_log_op_{args.tok_logit_operation}" if (args.detector_name_bis in ["GDINO_SwinB", "GDINO_SwinT"]) else ""
        )
        if args.detector_name not in ["Florence-2-large-v2", "Florence-2-large-ft-v2"]:
            filepath_step_2_bis = os.path.join(
                args.main_folder,
                args.dataset,
                args.detector_name_bis,
                "new_outputs_postpro_step_2",
                f"{data_name}_{args.split}_{args.min_samples}_{length}_samples_with_{args.step_1_method}"
                + articles
                + (
                    f"{log_op_str}_box_t_subject_{args.box_t_subject}_box_t_subquery_{args.box_t_subquery}"
                    f"_box_t_full_sent_{args.box_t_full_sent}_cxcywh_format.pt"
                ),
            )
            full_filepath_step_2_bis = os.path.join(
                args.main_folder,
                args.dataset,
                args.detector_name_bis,
                "new_outputs_postpro_step_2",
                f"{data_name}_{args.split}_0_{full_max_sample}_samples_with_{args.step_1_method}"
                + articles
                + (
                    f"{log_op_str}_box_t_subject_{args.box_t_subject}_box_t_subquery_{args.box_t_subquery}"
                    f"_box_t_full_sent_{args.box_t_full_sent}_cxcywh_format.pt"
                ),
            )
        elif args.detector_name in ["Florence-2-large-v2", "Florence-2-large-ft-v2"]:
            filepath_step_2_bis = os.path.join(
                args.main_folder,
                args.dataset,
                args.detector_name_bis,
                "new_outputs_postpro_step_2",
                f"{data_name}_{args.split}_{args.min_samples}_{length}_samples_cxcywh_format.pt",
            )
            full_filepath_step_2_bis = os.path.join(
                args.main_folder,
                args.dataset,
                args.detector_name_bis,
                "new_outputs_postpro_step_2",
                f"{data_name}_{args.split}_0_{full_max_sample}_samples_cxcywh_format.pt",
            )

        if not os.path.exists(full_filepath_step_2_bis):
            print(
                "\nfull_filepath_step_2_bis does not exist at:",
                full_filepath_step_2_bis,
            )

    prepro_data_filepaths = {
        "step_1": filepath_step_1,
        "full_step_1": full_filepath_step_1,
        "step_2": filepath_step_2,
        "full_step_2": full_filepath_step_2,
        "step_2_bis": filepath_step_2_bis,
        "full_step_2_bis": full_filepath_step_2_bis,
    }

    return prepro_data_filepaths


def define_short_LLM_folder_name(HF_LLM_name):

    ft_list = [
        "Llama3_8",
        "Llama3_70",
        "Mixtral_8x7",
        "Mixtral_8x22",
        "gemma-2-2b-it",
        "gemma-2-9b-it",
        "gpt-neo-2.7B",
        "gpt-neo-1.3B",
        "gpt-neo-125m",
    ]

    full_list = ft_list + ["Llama-3-8B", "Llama-3-70B", "Mixtral-8x7B", "Mixtral-8x22B"]

    found_llm = False
    for llm in full_list:
        if llm in HF_LLM_name:
            found_llm = True
    assert found_llm is True

    # if use llm not FT
    if HF_LLM_name == "Meta-Llama-3-8B-Instruct":
        short_name = "Llama3_8"
    elif HF_LLM_name == "Meta-Llama-3-70B-Instruct":
        short_name = "Llama3_70"
    elif HF_LLM_name == "Mixtral-8x7B-Instruct-v0.1":
        short_name = "Mixtral_8x7"
    elif HF_LLM_name == "Mixtral-8x22B-Instruct-v0.1":
        short_name = "Mixtral_8x22"
    elif HF_LLM_name == "gemma-2-9b-it":
        short_name = "gemma-2-9b-it"
    elif HF_LLM_name == "gemma-2-2b-it":
        short_name = "gemma-2-2b-it"
    elif HF_LLM_name in ["gpt-neo-2.7B", "gpt-neo-1.3B", "gpt-neo-125m"]:
        short_name = HF_LLM_name

    # if use FT llm
    else:
        for elem in ft_list:
            if elem in HF_LLM_name:
                short_name = elem

    return short_name


def define_FT_data_filename(args, split=None, max_samples=None):
    split = args.split if split is None else split

    if args.dataset == "talk2car":
        data_name = "talk2car"
        if max_samples is None:
            if split == "val":
                max_samples = 1163
            elif split == "test":
                max_samples = 2447
            elif split == "train":
                max_samples = 8348

    if args.dataset == "HC_RefLoCo":
        data_name = "HC_RefLoCo"
        assert split != "train"
        if max_samples is None:
            if split == "val":
                max_samples = 13360
            elif split == "test":
                max_samples = 31378

    if args.dataset == "RefCOCOg_umd":
        data_name = "Refcocog"
        if max_samples is None:
            if split == "val":
                max_samples = 4896
            elif split == "test":
                max_samples = 9602
            elif split == "train":
                max_samples = 80512

    elif args.dataset == "RefCOCO_unc":
        data_name = "Refcoco"
        if max_samples is None:
            if split == "val":
                max_samples = 10834
            elif split == "test":
                max_samples = 10752
            elif split == "train":
                max_samples = 120624
            elif split == "testA":
                max_samples = 5657
            elif split == "testB":
                max_samples = 5095

    elif args.dataset == "RefCOCO+_unc":
        data_name = "Refcoco+"
        if max_samples is None:
            if split == "val":
                max_samples = 10758
            elif split == "test":
                max_samples = 10615
            elif split == "train":
                max_samples = 120191
            elif split == "testA":
                max_samples = 5726
            elif split == "testB":
                max_samples = 4889

    assert isinstance(max_samples, int)

    max_nb_of_box_proposal_str = (
        "_max_box_" + str(args.max_nb_of_box_proposal) if args.max_nb_of_box_proposal is not None else ""
    )

    if "Florence-2" in args.detector_name:
        article_string = ""
        step_1_str = ""
        log_op_str = ""
        box_t_subj_str = ""
        box_t_subq_str = ""
        box_t_full_sent_str = ""
    else:
        article_string = "wo_articles_"
        step_1_str = f"_with_{args.step_1_method}"
        log_op_str = f"log_op_{args.tok_logit_operation}_"
        box_t_subj_str = f"_box_t_subj_{args.box_t_subject}"
        box_t_subq_str = f"_box_t_subq_{args.box_t_subquery}"
        box_t_full_sent_str = f"_box_t_full_sent_{args.box_t_full_sent}"

    FT_data_filename = (
        f"{data_name}_{split}_{args.min_samples}_{max_samples}_samples{step_1_str}"
        + f"_{args.prompt_context}{max_nb_of_box_proposal_str}_{article_string}{log_op_str}"
        + f"prompt_{args.prompt_version}{box_t_subj_str}{box_t_subq_str}{box_t_full_sent_str}_bbox_{args.bbox_format}"
    )

    return FT_data_filename


def get_FT_data_filename(my_args, for_FT):

    if for_FT:  # train data
        FT_data_filename = define_FT_data_filename(args=my_args, split="train", max_samples=my_args.Nb_train_data)

    else:  # val data
        if my_args.eval_on_train_batch:
            FT_data_filename = define_FT_data_filename(args=my_args, split="train", max_samples=my_args.Nb_val_data)
        else:
            FT_data_filename = define_FT_data_filename(args=my_args, split="val", max_samples=my_args.Nb_val_data)

    FT_data_filename += "_CLEAN.json"

    if for_FT:
        print("File used for FT:", FT_data_filename)
    else:
        print("File used for val:", FT_data_filename)

    return FT_data_filename


def define_FT_model_name(my_args, FT_data_filename):

    short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=my_args.HF_LLM_name)

    if "train" in FT_data_filename:
        split_used_to_FT = "train"
    elif "val" in FT_data_filename:
        split_used_to_FT = "val"
    else:
        split_used_to_FT = "test"

    if my_args.FT_data_modif == "aug":
        data_str = "_aug_data_"
    if my_args.FT_data_modif == "shuffle":
        data_str = "_shuffle_data_"
    else:
        data_str = "_data_"

    layer_str = "" if my_args.modules_to_FT is None else my_args.modules_to_FT + "_"

    end_string = split_used_to_FT + data_str + "pe_" + FT_data_filename.split("samples_")[1].split(".json")[0]
    this_detector = (
        my_args.detector_name if my_args.detector_name_bis is None else my_args.detector_name + "_" + my_args.detector_name_bis
    )

    if my_args.dataset == "RefCOCOg_umd":
        dataset_name_str = "Refg"
    elif my_args.dataset == "RefCOCO_unc":
        dataset_name_str = "Ref"
    elif my_args.dataset == "RefCOCO+_unc":
        dataset_name_str = "Ref+"
    elif my_args.dataset == "talk2car":
        dataset_name_str = "talk2car"
    else:
        raise Exception("Not implemented for this dataset")

    save_FT_model_name = (
        f"FT_{short_LLM_name}_{this_detector}_{layer_str}r_{my_args.LORA_r}_a_{my_args.LORA_alpha}"
        + f"_BS_{my_args.BS}_LR_{my_args.LR}_FT_on_{my_args.max_len_train}_{dataset_name_str}_"
        + end_string
    )
    save_FT_model_name = save_FT_model_name.replace("format_", "")

    if my_args.eval_on_train_batch:
        save_FT_model_name += f"_classic_EVAL_on_{my_args.max_len_train}_TRAIN_DATA"

    return save_FT_model_name


def define_model_result_name(args, checkpoint_nb=None):

    # to say at how many box we cut the prompt
    max_nb_of_box_proposal_str = (
        "_max_box_" + str(args.max_nb_of_box_proposal) if args.max_nb_of_box_proposal is not None else ""
    )

    log_op_str = f"log_op_{args.tok_logit_operation}" if (args.detector_name in ["GDINO_SwinB", "GDINO_SwinT"]) else ""

    if "GDINO_Swin" in args.detector_name:
        article_string = "wo_articles_"
        step_1_str = f"_with_{args.step_1_method}"
        box_t_subj_str = f"_box_t_subj_{args.box_t_subject}"
        box_t_subq_str = f"_box_t_subq_{args.box_t_subquery}"
        box_t_full_sent_str = f"_box_t_full_sent_{args.box_t_full_sent}"

    else:
        article_string = ""
        step_1_str = ""
        box_t_subj_str = ""
        box_t_subq_str = ""
        box_t_full_sent_str = ""

    # VLM only
    if args.VLM_only:
        if args.detector_name not in ["GDINO_SwinT", "GDINO_SwinB"]:
            if not args.limit_to_300_samples_GDINO_1_5:
                if not args.spe_fall_back_for_vlm_0s:
                    model_name_to_use = (
                        f"{args.detector_name}_only_{args.split}_{args.min_samples}"
                        f"_{args.max_samples}{box_t_full_sent_str}.json"
                    )
                else:
                    model_name_to_use = (
                        f"{args.detector_name}_+_fallback_strat_{args.split}_{args.min_samples}"
                        f"_{args.max_samples}{box_t_full_sent_str}.json"
                    )
            else:
                if "RefCOCOg" not in args.dataset:
                    raise Exception("not possible")
                if not args.spe_fall_back_for_vlm_0s:
                    model_name_to_use = f"{args.detector_name}_only_on_300_val_data_of_GDINO_1_5{box_t_full_sent_str}.json"
                else:
                    model_name_to_use = (
                        f"{args.detector_name}_only_on_300_val_data_of_GDINO_1_5_+_FALLBACK_strat{box_t_full_sent_str}.json"
                    )

        elif args.detector_name in ["GDINO_SwinT", "GDINO_SwinB"]:
            model_name_to_use = f"{args.detector_name}_only_with_subj_tok_scores_{article_string}{log_op_str}"
            if not args.limit_to_300_samples_GDINO_1_5:
                model_name_to_use += (
                    f"_{args.split}_{args.min_samples}_"
                    f"{args.max_samples}{box_t_subj_str}{box_t_subq_str}{box_t_full_sent_str}.json"
                )
            else:
                model_name_to_use += f"_on_300_val_data_of_GDINO_1_5{box_t_subj_str}{box_t_subq_str}{box_t_full_sent_str}.json"

    # Oracle
    elif args.prompt_version == 0:
        model_name_to_use = (
            f"Oracle_{args.detector_name}_pe_{args.prompt_context}{max_nb_of_box_proposal_str}"
            f"_{args.split}_{args.min_samples}_{args.max_samples}{box_t_subj_str}{box_t_subq_str}{box_t_full_sent_str}.json"
        )

    # LLM use
    else:

        if args.folder_with_checkpoints_to_eval is not None:
            model_name_to_use = checkpoint_nb + ".json"

        else:
            if args.dataset == "RefCOCOg_umd":
                data_name = "Refcocog"
            elif args.dataset == "RefCOCO_unc":
                data_name = "Refcoco"
            elif args.dataset == "RefCOCO+_unc":
                data_name = "Refcoco+"
            elif args.dataset == "talk2car":
                data_name = "talk2car"

            # Base LLM (not finetuned)
            if args.HF_LLM_name in [
                "Mixtral-8x7B-Instruct-v0.1",
                "Mixtral-8x22B-Instruct-v0.1",
                "Meta-Llama-3-8B-Instruct",
                "Meta-Llama-3-70B-Instruct",
                "gemma-2-2b-it",
                "gemma-2-9b-it",
                "gpt-neo-2.7B",
                "gpt-neo-1.3B",
                "gpt-neo-125m",
            ]:
                short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=args.HF_LLM_name)

                if not args.limit_to_300_samples_GDINO_1_5:
                    model_name_to_use = (
                        f"HF_ori_{short_LLM_name}_{data_name}_{args.split}_{args.min_samples}"
                        + f"_{args.max_samples}_samples{step_1_str}"
                        + f"_pe_{args.prompt_context}{max_nb_of_box_proposal_str}"
                        + f"_{article_string}{log_op_str}_prompt_{args.prompt_version}"
                    )
                else:
                    model_name_to_use = (
                        f"HF_ori_{short_LLM_name}_{data_name}_on_300_val_data_of_GDINO_1_5{step_1_str}"
                        + f"_pe_{args.prompt_context}{max_nb_of_box_proposal_str}"
                        + f"_{article_string}{log_op_str}_prompt_{args.prompt_version}"
                    )

                model_name_to_use += f"{box_t_subj_str}{box_t_subq_str}{box_t_full_sent_str}_bbox_{args.bbox_format}.json"

            # Finetuned LLM
            else:
                model_name_to_use = (
                    args.HF_LLM_name.replace("checkpoint", "ckpt")
                    .replace("subquery", "subq")
                    .replace("CLEAN_auto_saved_", "")
                    .replace("prompt", "p")
                    .replace("log_op_", "")
                    .replace("articles", "art")
                    .replace("/", "_")
                    .replace("subject_", "subj_")
                    .replace("box_t_", "")
                    .replace("format_", "")
                )
                if not args.limit_to_300_samples_GDINO_1_5:
                    model_name_to_use += (
                        f"_on_{args.split}_{args.min_samples}_{args.max_samples}_"
                        f"pe_{args.prompt_context}{max_nb_of_box_proposal_str}.json"
                    )
                else:
                    model_name_to_use += (
                        f"_on_300_val_data_of_GDINO_1_5_pe_{args.prompt_context}{max_nb_of_box_proposal_str}.json"
                    )

    return model_name_to_use


def get_checkpoint_list_and_paths_for_eval(args):

    # 1) get name of folder containing checkpoints using given args
    if str(args.folder_with_checkpoints_to_eval) == "auto":

        FT_train_data_filename = get_FT_data_filename(my_args=args, for_FT=True)
        save_FT_model_name = define_FT_model_name(my_args=args, FT_data_filename=FT_train_data_filename)
        folder_with_checkpoints = save_FT_model_name + "_auto_saved"
        print(f"Auto: Loading LoRA weights from folder {folder_with_checkpoints}")
    # use specified folder using folder name
    else:
        folder_with_checkpoints = args.folder_with_checkpoints_to_eval

    # 2) define full path to folder
    path_to_folder = os.path.join(args.main_folder, args.dataset, "my_FT_models", folder_with_checkpoints)
    if os.path.exists(path_to_folder):
        print(f"checkpoint folder {folder_with_checkpoints} exists")
    else:
        raise Exception(f"in my_FT_models folder, there is no {folder_with_checkpoints}")

    # 3) get the list of all checkpoint subfolders to evaluate

    # list subfolders in checkpoint folder
    checkpoint_subfolders = [f for f in os.listdir(path_to_folder) if "checkpoint" in f]
    print(
        f"\n{len(checkpoint_subfolders)} checkpoints in total in folder",
        folder_with_checkpoints,
    )

    # list those for whom we have already done an evaluation
    short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=args.HF_LLM_name)
    result_folder = os.path.join(
        args.main_folder,
        args.dataset,
        args.detector_name,
        short_LLM_name,
        "results",
        folder_with_checkpoints,
    )

    if os.path.exists(result_folder):
        print(f"folder '{result_folder}' already exists")
        checkpoint_subfolders_already_eval = [f.replace(".json", "") for f in os.listdir(result_folder) if "checkpoint" in f]
        checkpoint_subfolders = [elem for elem in checkpoint_subfolders if elem not in checkpoint_subfolders_already_eval]

    # reorder checkpoints in my list
    checkpoint_subfolders_int = [int(e.replace("checkpoint-", "")) for e in checkpoint_subfolders]
    if args.exact_checkpt_list is not None:
        checkpoint_subfolders_int = [e for e in args.exact_checkpt_list if e in checkpoint_subfolders_int]
    elif args.last_checkpt:
        checkpoint_subfolders_int = [max([e for e in checkpoint_subfolders_int])]
    else:
        checkpoint_subfolders_int = [
            e for e in checkpoint_subfolders_int if e >= args.min_checkpt and e <= args.max_checkpt
        ]  # cut if too many checkpoints
        checkpoint_subfolders_int.sort()

    if args.ckpt_multiple is not None:
        checkpoint_subfolders_int = [e for e in checkpoint_subfolders_int if e % args.ckpt_multiple == 0]

    checkpoint_subfolders = ["checkpoint-" + str(e) for e in checkpoint_subfolders_int]
    print(
        f"{len(checkpoint_subfolders)} checkpoints not yet evaluated:\n",
        checkpoint_subfolders,
    )

    return path_to_folder, folder_with_checkpoints, checkpoint_subfolders
