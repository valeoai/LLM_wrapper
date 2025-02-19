import json
import os
import random
import time

import torch

# Functions for LLM loading


def get_HF_model_id(HF_LLM_name):

    if "Mixtral_8x7" in HF_LLM_name or "Mixtral-8x7" in HF_LLM_name:
        model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        base_model_name = "Mixtral-8x7B-Instruct-v0.1"

    elif "Mixtral_8x22" in HF_LLM_name or "Mixtral-8x22" in HF_LLM_name:
        model_id = "mistralai/Mixtral-8x22B-Instruct-v0.1"
        base_model_name = "Mixtral-8x22B-Instruct-v0.1"

    elif "Llama-3-8B" in HF_LLM_name or "Llama3_8" in HF_LLM_name:
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        base_model_name = "Meta-Llama-3-8B-Instruct"

    elif "Llama-3-70B" in HF_LLM_name or "Llama3_70" in HF_LLM_name:
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        base_model_name = "Meta-Llama-3-70B-Instruct"

    elif "gemma-2-9b-it" in HF_LLM_name:
        model_id = "google/gemma-2-9b-it"
        base_model_name = "gemma-2-9b-it"

    elif "gemma-2-2b-it" in HF_LLM_name:
        model_id = "google/gemma-2-2b-it"
        base_model_name = "gemma-2-2b-it"

    elif "gpt-neo-2.7B" in HF_LLM_name:
        model_id = "EleutherAI/gpt-neo-2.7B"
        base_model_name = "gpt-neo-2.7B"

    elif "gpt-neo-1.3B" in HF_LLM_name:
        model_id = "EleutherAI/gpt-neo-1.3B"
        base_model_name = "gpt-neo-1.3B"

    elif "gpt-neo-125m" in HF_LLM_name:
        model_id = "EleutherAI/gpt-neo-125m"
        base_model_name = "gpt-neo-125m"

    else:
        raise Exception(f"not implemented for the LLM ({HF_LLM_name})")

    return model_id, base_model_name


def load_HF_model_tok(args, HF_LLM_name, eval_mode=True, FT_mode=False, timing=True, quantization=True):

    if timing:
        s = time.time()

    with open(args.HF_token_path) as f:
        HF_TOKEN = json.load(f)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if HF_LLM_name == "Mixtral-8x22B-Instruct-v0.1":
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch  # noqa: F401

    # Find base model
    model_id, base_model_name = get_HF_model_id(HF_LLM_name=HF_LLM_name)

    # Define model / tokenizer's cache folders
    cache_dir = os.path.join(args.main_folder, "LLM_cache", f"cache_{base_model_name}")
    cache_dir_tok = os.path.join(args.main_folder, "LLM_cache", f"cache_{base_model_name}_tokenizer")
    print(f"(Down)loading base model {base_model_name} from/to {args.main_folder}/LLM_cache/")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir_tok, token=HF_TOKEN, force_download=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    use_cache = True if FT_mode is False else False
    attn_implem = "eager" if "gemma" in HF_LLM_name else "flash_attention_2"

    if quantization:

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            quantization_config=nf4_config,
            use_cache=use_cache,
            attn_implementation=attn_implem,
            token=HF_TOKEN,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            # quantization_config=nf4_config,
            use_cache=use_cache,
            attn_implementation=attn_implem,
            token=HF_TOKEN,
        )

    # add FT lora weights to base model
    if HF_LLM_name not in [
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

        # First loading option => instance directly FT model but it redownloads shards/base model
        """
        model = AutoModelForCausalLM.from_pretrained(FT_model_save_path,
                                                    device_map='auto',
                                                    quantization_config=nf4_config,
                                                    use_cache=use_cache,
                                                    attn_implementation=attn_implem,
                                                    token=HF_TOKEN)
        """

        # Second loading option => instance FT model efficiently on top of base model
        ft_dataset = args.ft_dataset if args.ft_dataset is not None else args.dataset
        path_to_folder = os.path.join(args.main_folder, ft_dataset, "my_FT_models")
        assert os.path.exists(os.path.join(path_to_folder, HF_LLM_name))
        model = load_FT_model_via_base_model(
            base_model=model,
            complete_FT_path=os.path.join(path_to_folder, HF_LLM_name),
            timing=timing,
        )

    if eval_mode:
        model.eval()

    if timing:
        e = time.time()
        print("HF LLM / tokenizer loading time:", round(e - s, 1), "secs")

    return model, tokenizer


def load_FT_model_via_base_model(base_model, complete_FT_path, timing):
    if timing:
        s = time.time()

    # new loading way (load base model + give checkpoint path)
    from peft import PeftModel

    print(f"add lora adapters to base model from: {complete_FT_path}")
    FT_model = PeftModel.from_pretrained(base_model, complete_FT_path)
    if timing:
        e = time.time()
        print("Time to load lora modules on base model:", round(e - s, 2), "secs")
    return FT_model


# Functions for text generation


def gen_step_3_with_HF(HF_LLM_name, model, tokenizer, prompt, verbose=False, max_gen_tok=None):

    encoded_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    model_inputs = encoded_input.to("cuda")
    max_new_tokens = 128 if max_gen_tok is None else max_gen_tok

    if "Mixtral" in HF_LLM_name or "gemma" in HF_LLM_name or "gpt-neo" in HF_LLM_name:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    elif "Llama" in HF_LLM_name or "llama" in HF_LLM_name:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    stripped_result = tokenizer.batch_decode(generated_ids[0][encoded_input["input_ids"][0].shape[0] :].unsqueeze(0))[0]
    if "Mixtral" in HF_LLM_name:
        stripped_result = stripped_result.replace("</s>", "").strip()
    elif "Llama" in HF_LLM_name or "llama" in HF_LLM_name:
        stripped_result = stripped_result.replace("<|eot_id|>", "").strip()
    elif "gemma" in HF_LLM_name:
        if "gemma-2-2b-it" in HF_LLM_name:
            print("gemma : before strip:", stripped_result)
        stripped_result = (
            stripped_result.strip("<eos>").strip().strip("\n").strip().strip("<end_of_turn>").strip().strip("\n").strip()
        )
        if "gemma-2-2b-it" in HF_LLM_name:
            print("gemma : after strip:", stripped_result)
    elif "gpt-neo" in HF_LLM_name:
        stripped_result = stripped_result.strip()
    else:
        print("WARNING stripping on generation by this LLM not implemented")

    if verbose:
        print("LLM generated (stripped) result:", stripped_result)

    return stripped_result


def extract_int_result_from_LLM_gen(result, HF_LLM_name, return_error_type, nb_detected_boxes=None, fixed_seed=None):

    # Clean up specific to gemma-2-2b
    result = result.strip().strip("'").strip('"')
    if HF_LLM_name == "gemma-2-2b-it":
        for i in range(9):
            result = result.replace(str(i) + ". ", "")

    # Int extraction
    new_result = ""
    started_extraction = False
    for char in result:
        if char in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            new_result += char
            started_extraction = True
        else:
            if started_extraction is True:
                break

    # Error analysis if specified
    if return_error_type:

        error_type = None  # indicate type of error if any

        if new_result != "":

            new_result = int(new_result)

            if new_result >= nb_detected_boxes:
                new_result = None
                error_type = "gen_out_of_scope"
            else:
                if fixed_seed is not None:
                    list_id = list(range(nb_detected_boxes))
                    random.seed(fixed_seed)
                    random.shuffle(list_id)
                    new_result = list_id[new_result]

        else:
            new_result = None
            error_type = "no_int_generated"

        return new_result, error_type
    else:
        return new_result
