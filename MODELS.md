# Models

All our models are released under a research-only [RAIL Model License](LICENSE_MODEL).

## Downloading pretrained LoRA weights

The models evaluated in our main results, using Florence-2-L as VLM, are released on our GitHub.

For Llama 3 8B and Mixtral 8x7B, we use the following script to convert the weights to tar files:

```bash
# Performs something akin to:
# tar czf - filename | split -b 1900MB - filename.tar.gz.part_
python handle_checkpoints.py \
--mode create \
--checkpoint_path XXXX \
--outdir llm_wrapper_release \
--maxsize 1900MB
```

All versions of Mixtral 8x7B's weights are converted into a single tar file. You can untar them with the simple command:

```bash
tar -xvzf filename.tar.gz
```

All versions of Llama 3 8B's weights are chunked into three tar files. You can merge them using the following command:

1. Download all tar files (for Llama 3 8B and a chosen training data).
2. Put them in a single folder (e.g., `FT_Llama3_8_on_RefCOCOg_for_Flo2L_chunks`).
3. Run the following command:

```bash
# Performs something akin to:
# cat filename.tar.gz.part_* > filename.tar.gz
# tar xzf filename.tar.gz
python handle_checkpoints.py \
--mode extract \
--checkpoint_path FT_Llama3_8_on_RefCOCOg_for_Flo2L_chunks \
--outdir XXXX
```

Once done, as mentioned in our [README.md](README.md), place the untared checkpoint folder in a subfolder named ``my_FT_models``, placed in the proper folder, depending on the chosen training data (``./llm_wrapper/data/{dataset_name}/my_FT_models/{checkpoint_folder}``).

