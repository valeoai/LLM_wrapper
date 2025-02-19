import json
import os
import time

import pandas as pd
import torch
from datasets import Dataset
from LLM_utils import extract_int_result_from_LLM_gen, gen_step_3_with_HF, load_HF_model_tok
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

from llm_wrapper.config_llm_wrapper import get_args
from llm_wrapper.naming_utils import define_FT_model_name, define_short_LLM_folder_name, get_FT_data_filename

# ----------------------------------------------------------------------
# Functions to get the finetuning data in the right format
# ----------------------------------------------------------------------


def create_my_prompt(sample):
    # balises "<s>"" and "</s>" are automatically added by trainer it seems
    full_prompt = sample["prompt"].replace("<s> ", "")
    full_prompt += sample["best_subject_index"]
    return full_prompt


def create_my_datasetdict(FT_data_filename, FT_data_folder, max_len=None, verbose=True):

    if verbose:
        s = time.time()

    with open(os.path.join(FT_data_folder, FT_data_filename)) as f:
        prompt_dataset = json.load(f)

    # Create a list of texts with the right prompt format
    if isinstance(prompt_dataset, list):
        X = [create_my_prompt(val) for val in prompt_dataset if val != {}]
    elif isinstance(prompt_dataset, dict):
        X = [create_my_prompt(val) for val in prompt_dataset.values()]
    X = X[:max_len] if max_len is not None else X

    # Make a dict then a dataframe then a DatasetDict from it
    X_dict = {"text": X}
    df = pd.DataFrame(X_dict)
    my_dataset = Dataset.from_pandas(df)

    if verbose:
        e = time.time()
        print(f"Built fine-tuning dataset of len {len(my_dataset)} in {round(e-s, 1)} secs")

    return my_dataset


# ----------------------------------------------------------------------
# Function specifying the LLM modules to finetune with LoRA
# ----------------------------------------------------------------------


def define_modules_to_FT(my_args):
    # Set the modules to apply the adapter to.
    # If specified in peft_config, only the modules with the specified names will be replaced.
    if my_args.modules_to_FT is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]

    elif my_args.modules_to_FT == "lm_head_only":
        target_modules = [
            "lm_head",
        ]

    elif my_args.modules_to_FT == "last_4_only":
        target_modules = [
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]

    return target_modules


# ----------------------------------------------------------------------
# Functions to use callback (optional, only if use --use_callbacks)
# ----------------------------------------------------------------------


class PrintCallback(TrainerCallback):
    def __init__(
        self,
        my_args,
        HF_LLM_name,
        writer,
        good_bad_idx_dict,
        my_TRAIN_dataloader,
        my_VAL_dataloader,
        max_eval_data,
        save_FT_model_name,
    ):
        self.my_TRAIN_dataloader = my_TRAIN_dataloader
        self.my_VAL_dataloader = my_VAL_dataloader
        self.save_FT_model_name = save_FT_model_name
        self.last_eval_epoch = 0
        self.max_eval_data = max_eval_data
        self.writer = writer
        self.good_bad_idx_dict = good_bad_idx_dict
        self.HF_LLM_name = HF_LLM_name
        self.my_args = my_args
        self.use_callbacks = my_args.use_callbacks

    def on_step_end(self, args, state, control, **kwargs):

        if state.global_step % 500 == 0:

            if self.use_callbacks and "gemma" not in self.HF_LLM_name:
                s = time.time()
                with torch.cuda.amp.autocast():
                    our_eval(
                        my_args=self.my_args,
                        HF_LLM_name=self.HF_LLM_name,
                        model=kwargs["model"],
                        tokenizer=kwargs["tokenizer"],
                        writer=self.writer,
                        my_TRAIN_dataloader=self.my_TRAIN_dataloader,
                        my_VAL_dataloader=self.my_VAL_dataloader,
                        max_eval_data=self.max_eval_data,
                        save_FT_model_name=self.save_FT_model_name + f"_OUR_EVAL_on_{self.max_eval_data}_train&val_data",
                        global_epoch=state.global_step,
                        verbose=True,
                        find_idx=False,
                        good_bad_idx_dict=self.good_bad_idx_dict,
                    )

                print("Time for callback:", round(time.time() - s, 2), "secs")

            else:
                # print("\nEpoch:", int(state.epoch), " | Iter:", state.global_step)
                self.writer.add_scalar("Time wrt step", time.time(), state.global_step)
                print(f"Step {state.global_step} | current time:", round(time.time(), 2))

        return None


def our_eval(
    my_args,
    HF_LLM_name,
    model,
    tokenizer,
    writer,
    my_TRAIN_dataloader,
    my_VAL_dataloader,
    max_eval_data,
    save_FT_model_name,
    global_epoch=None,
    verbose=True,
    find_idx=False,
    nb_good_idx=5,
    good_bad_idx_dict=None,
):

    if "Llama" in HF_LLM_name or "llama" in HF_LLM_name:
        stop_string = "assistant<|end_header_id|>"
    elif "Mixtral" in HF_LLM_name:
        stop_string = "[/INST]"
    elif "gemma" in HF_LLM_name:
        stop_string = "<start_of_turn>model\n"
    elif "gpt" in HF_LLM_name:
        stop_string = "\nAnswer:"
    else:
        raise Exception("stop string not implemented for given LLM")

    # print("Using our_eval at epoch n°", global_epoch)
    short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=my_args.HF_LLM_name)
    this_detector = (
        my_args.detector_name if my_args.detector_name_bis is None else my_args.detector_name + "_" + my_args.detector_name_bis
    )
    folder = os.path.join(
        my_args.main_folder,
        my_args.dataset,
        this_detector,
        short_LLM_name,
        "FT_eval",
        f"{max_eval_data}_eval_data",
    )
    fp = os.path.join(folder, save_FT_model_name + ".json")

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    if os.path.exists(fp):
        with open(fp) as f:
            dict_nb_data_found = json.load(f)
        # case eval was already done (if resume training for instance)
        if str(global_epoch) in dict_nb_data_found["train"] and str(global_epoch) in dict_nb_data_found["val"]:
            if find_idx:
                if "good_bad_idx_dict" in dict_nb_data_found:
                    if (
                        "train_good_idx" in dict_nb_data_found["good_bad_idx_dict"]
                        and "val_good_idx" in dict_nb_data_found["good_bad_idx_dict"]
                    ):
                        if (
                            "train_bad_idx" in dict_nb_data_found["good_bad_idx_dict"]
                            and "val_bad_idx" in dict_nb_data_found["good_bad_idx_dict"]
                        ):
                            print("we already did the index search and eval for step=0")
                            for key in dict_nb_data_found["good_bad_idx_dict"].keys():
                                print(
                                    "for key ",
                                    key,
                                    " | len idx :",
                                    len(dict_nb_data_found["good_bad_idx_dict"][key]),
                                )
                            return dict_nb_data_found["good_bad_idx_dict"]
            else:
                print("we already have numbers for step", str(global_epoch))
                return None

    else:
        dict_nb_data_found = {"train": {}, "val": {}}
        dict_nb_data_found["max_len_train"] = len(my_TRAIN_dataloader)
        dict_nb_data_found["max_len_val"] = len(my_VAL_dataloader)
        dict_nb_data_found["max_eval_data"] = max_eval_data

    if find_idx:
        good_bad_idx_dict = {}

    for loader_type, one_loader in zip(["train", "val"], [my_TRAIN_dataloader, my_VAL_dataloader]):
        cnt, not_clean = 0, 0
        used_it = 0
        if find_idx:
            good_idx_list, bad_idx_list = [], []

        for it, elem in enumerate(one_loader):
            if find_idx:
                if len(good_idx_list) == nb_good_idx and len(bad_idx_list) == (max_eval_data - nb_good_idx):
                    print(f"all good / bad eval idx found for {loader_type} data")
                    good_bad_idx_dict[f"{loader_type}_good_idx"] = good_idx_list
                    good_bad_idx_dict[f"{loader_type}_bad_idx"] = bad_idx_list
                    break
            elif not find_idx:
                if used_it == max_eval_data:
                    print("used_it == max_eval_data")
                    break
                if it in good_bad_idx_dict[f"{loader_type}_good_idx"] or it in good_bad_idx_dict[f"{loader_type}_bad_idx"]:
                    used_it += 1
                else:
                    continue

            prompt = elem["text"][0].split(stop_string)[0] + stop_string.strip()
            expected_answer = elem["text"][0].split(stop_string)[1].strip()

            gen_answer = gen_step_3_with_HF(
                HF_LLM_name=HF_LLM_name,
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                verbose=False,
            )
            clean_gen_answer = extract_int_result_from_LLM_gen(
                result=gen_answer, HF_LLM_name=HF_LLM_name, return_error_type=False
            )
            if verbose:
                print(
                    f"{loader_type} data n°{it} | GT:",
                    expected_answer,
                    "| clean gen:",
                    clean_gen_answer,
                )

            try:
                if int(clean_gen_answer) == int(expected_answer):
                    cnt += 1
                    if find_idx and len(good_idx_list) < nb_good_idx:
                        good_idx_list.append(it)
                else:
                    if find_idx and len(bad_idx_list) < (max_eval_data - nb_good_idx):
                        bad_idx_list.append(it)
            except:
                print("gen:", gen_answer)
                not_clean += 1

        if find_idx:
            prop_data_found = nb_good_idx / max_eval_data
            prop_unclean_data = 0
        elif not find_idx:
            prop_data_found = cnt / max_eval_data
            prop_unclean_data = not_clean / max_eval_data

        print(f"\nProp {loader_type} data found: {prop_data_found} {loader_type} data")
        print(f"Prop unclean gen: {prop_unclean_data} {loader_type} data")
        dict_nb_data_found[loader_type][global_epoch] = {
            "%_right_int_gen": prop_data_found,
            "%_unclean_gen": prop_unclean_data,
        }

        if writer is not None and global_epoch is not None:
            writer.add_scalar(
                f"Our_eval/%_right_int_gen_{max_eval_data}_{loader_type}",
                prop_data_found,
                global_epoch,
            )
            writer.add_scalar(
                f"Our_eval/%_unclean_gen_{max_eval_data}_{loader_type}",
                prop_unclean_data,
                global_epoch,
            )

    if find_idx:
        for key in good_bad_idx_dict.keys():
            print("for key ", key, " | len idx :", len(good_bad_idx_dict[key]))
        dict_nb_data_found["good_bad_idx_dict"] = good_bad_idx_dict

    if global_epoch is not None:
        with open(fp, "w") as outfile:
            json.dump(dict_nb_data_found, outfile, indent=4)

    if find_idx:
        if len(good_idx_list) != nb_good_idx or len(bad_idx_list) != (max_eval_data - nb_good_idx):
            raise Exception("Not enough good/bad idx found")
        return good_bad_idx_dict
    else:
        return None


# ---------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------


def main(my_args):

    # Print important args

    this_detector = (
        my_args.detector_name if my_args.detector_name_bis is None else my_args.detector_name + "_" + my_args.detector_name_bis
    )
    short_LLM_name = define_short_LLM_folder_name(HF_LLM_name=my_args.HF_LLM_name)
    print(f"\n\nAdapting (fine-tuning) VLM ({this_detector}) with LLM ({short_LLM_name})")
    print(f"Params:  epoch={my_args.epoch}  | BS = {my_args.BS} | LR = {my_args.LR}")
    print(f"LoRA params: r={my_args.LORA_r} | alpha={my_args.LORA_alpha}")

    # Define finetuning datasets

    FT_data_folder = os.path.join(
        my_args.main_folder,
        my_args.dataset,
        this_detector,
        short_LLM_name,
        "prompt_dataset_for_finetune",
        my_args.FT_data_modif,
    )

    FT_val_data_filename = get_FT_data_filename(my_args=my_args, for_FT=False)
    my_VAL_dataset = create_my_datasetdict(
        FT_data_filename=FT_val_data_filename,
        FT_data_folder=FT_data_folder,
        max_len=my_args.max_len_val,
    )
    print("len val dataset:", len(my_VAL_dataset))

    FT_train_data_filename = get_FT_data_filename(my_args=my_args, for_FT=True)
    my_TRAIN_dataset = create_my_datasetdict(
        FT_data_filename=FT_train_data_filename,
        FT_data_folder=FT_data_folder,
        max_len=my_args.max_len_train,
    )
    print("len train dataset:", len(my_TRAIN_dataset))

    my_TRAIN_dataloader = DataLoader(
        my_TRAIN_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=my_args.num_workers,
        pin_memory=True,
    )
    my_VAL_dataloader = DataLoader(
        my_VAL_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=my_args.num_workers,
        pin_memory=True,
    )

    # Define filepaths where to save checkpoints / tensorboards data
    # (test first if output dir already exists
    # (it must exist if --resume_from_checkpoint, it mustn't exist if not --resume_from_checkpoint))

    save_FT_model_name = define_FT_model_name(my_args=my_args, FT_data_filename=FT_train_data_filename)
    writer = SummaryWriter(
        log_dir=os.path.join(
            my_args.main_folder,
            my_args.dataset,
            "my_FT_models",
            "tensorboards",
            save_FT_model_name,
        )
    )
    output_dir = os.path.join(
        my_args.main_folder,
        my_args.dataset,
        "my_FT_models",
        save_FT_model_name + "_auto_saved",
    )
    if my_args.resume_from_checkpoint and not os.path.exists(output_dir):
        raise Exception("Cannot resume as output_dir does not exist at:", output_dir)
    if not my_args.resume_from_checkpoint and os.path.exists(output_dir):
        raise Exception("Output dir already exists, cannot overwrite it. Output dir:\n", output_dir)
    print("output_dir already exists ?", os.path.exists(output_dir))

    # Set training config

    peft_config = LoraConfig(
        lora_alpha=my_args.LORA_alpha,  # The alpha parameter for Lora scaling
        lora_dropout=0.1,  # The dropout probability for Lora layers
        r=my_args.LORA_r,  # Lora attention dimension (the “rank”).
        # Can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’,
        # the corresponding biases will be updated during training.
        bias="none",
        # The names of the modules to apply the adapter to.
        # If this is specified, only the modules with the specified names will be replaced.
        target_modules=define_modules_to_FT(my_args=my_args),
        task_type="CAUSAL_LM",
    )

    args = TrainingArguments(
        disable_tqdm=False,
        output_dir=output_dir,  # where to save preds and checkpoints
        num_train_epochs=my_args.epoch,
        # max_steps = 500, # comment out this line if you want to train in epochs
        per_device_train_batch_size=my_args.BS,
        per_device_eval_batch_size=my_args.BS,
        # warmup_steps = 0.03, # error
        logging_first_step=True,
        logging_strategy="steps",
        logging_steps=500,
        save_strategy=my_args.ckpts_save_strategy,  # choices=["steps", "no", "epoch"]
        save_steps=my_args.ckpt_save_steps,  # needs to be multiple of eval_steps if load_best_model_at_end == True
        # do_predict =  #
        # if "steps" as not "no" this sets do_eval to True / with "steps".
        # Evaluation is done (and logged) every eval_steps. (Also "epoch")
        evaluation_strategy="no",
        eval_steps=500,  # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=my_args.LR,
        bf16=True,
        # lr_scheduler_type='constant',
        # save_total_limit = 100,
        # load_best_model_at_end = True,
        metric_for_best_model="loss",
        greater_is_better=False,  # use False if use eval loss metric, True if use accuracy or similar
        logging_dir=os.path.join(
            my_args.main_folder,
            my_args.dataset,
            "my_FT_models",
            "tensorboards",
            f"{save_FT_model_name}",
        ),  # (str, optional) – Tensorboard log directory. Will default to runs/**CURRENT_DATETIME_HOSTNAME**.
        # seed = # default to 42
        dataloader_num_workers=my_args.num_workers,  # (int, optional, defaults to 0)–
        gradient_accumulation_steps=1,
    )

    # Define LLM to fine-tune (and use a first callback eval if specified)

    model, tokenizer = load_HF_model_tok(
        args=my_args,
        HF_LLM_name=my_args.HF_LLM_name,
        eval_mode=False,
        FT_mode=True,
        timing=True,
    )

    good_bad_idx_dict = None
    if my_args.use_callbacks and "gemma" not in my_args.HF_LLM_name:
        print("\nOur eval on zero shot LLM")
        s = time.time()
        good_bad_idx_dict = our_eval(
            my_args=my_args,
            HF_LLM_name=my_args.HF_LLM_name,
            model=model,
            tokenizer=tokenizer,
            writer=writer,
            my_TRAIN_dataloader=my_TRAIN_dataloader,
            my_VAL_dataloader=my_VAL_dataloader,
            max_eval_data=my_args.max_eval_data,
            save_FT_model_name=save_FT_model_name + f"_OUR_EVAL_on_{my_args.max_eval_data}_train&val_data",
            global_epoch=0,
            verbose=True,
            find_idx=True,
            nb_good_idx=my_args.nb_good_idx,
        )
        e = time.time()
        print(
            "Eval time (for 0 shot with index finding process):",
            round(e - s, 2),
            "secs",
        )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    # Setting up collator to fine-tune on prompt completion only

    if "Llama" in my_args.HF_LLM_name:
        response_template_string = "assistant<|end_header_id|>"
    elif "Mixtral" in my_args.HF_LLM_name:
        response_template_string = "Answer: [/INST]"
    elif "gemma" in my_args.HF_LLM_name:
        response_template_string = "<start_of_turn>model\n"
    elif "gpt" in my_args.HF_LLM_name:
        response_template_string = "\nAnswer:"
    response_template_ids = tokenizer.encode(response_template_string, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(  # instruction_template="<s> [INST]",
        response_template=response_template_ids, tokenizer=tokenizer
    )

    # Setting up trainer

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,  # Max number of input tokens given to the LLM
        tokenizer=tokenizer,
        # packing = True,
        # formatting_func = create_prompt, # this will aplly the create_prompt mapping to all training and test dataset
        args=args,
        train_dataset=my_TRAIN_dataset,
        eval_dataset=my_VAL_dataset,
        data_collator=collator,  # to use DataCollatorForCompletionOnlyLM
        packing=False,  # to use DataCollatorForCompletionOnlyLM
        callbacks=[
            PrintCallback(
                my_args,
                my_args.HF_LLM_name,
                writer,
                good_bad_idx_dict,
                my_TRAIN_dataloader,
                my_VAL_dataloader,
                my_args.max_eval_data,
                save_FT_model_name,
            )
        ],
    )

    # Train

    if my_args.resume_from_checkpoint:
        print(f"\nFT from existing checkpoint in {output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f"\nFT from base model {my_args.HF_LLM_name}")
        trainer.train()

    return None


if __name__ == "__main__":

    my_args = get_args()
    main(my_args)
