"""
Example usage:

1. Create tar files from checkpoints

python scripts/handle_checkpoints.py \
--mode create \
--checkpoint_dir /path/to/checkpoint \
--outdir weights/release \
--maxsize 2G

2. Extract tar files

python scripts/handle_checkpoints.py \
--mode extract \
--checkpoint_dir /path/to/checkpoint \
--outdir tmp/release
"""

import argparse
import os
import subprocess
from glob import glob


def create_tar_file(checkpoint_path: str, outdir: str, maxsize: str) -> None:
    file_path = os.path.basename(checkpoint_path)
    file_dir = os.path.dirname(checkpoint_path)
    filename = os.path.basename(checkpoint_path).split(".")[0]
    outdir = os.path.join(outdir, filename)
    os.makedirs(outdir, exist_ok=True)
    part_name = os.path.join(outdir, f"{filename}_chunked.tar.gz.part_")
    subprocess.run(f"tar czf - -C {file_dir} {file_path} | split -b {maxsize} - {part_name}", shell=True)

    # check whether a unique tar file is created
    tar_files = glob(f"{part_name}*")
    if len(tar_files) > 1:
        return

    os.rename(tar_files[0], os.path.join(outdir, f"{filename}.tar.gz"))


def extract_tar_file(tar_dir: str, outdir: str) -> None:
    # Concatenate tar parts
    tmpdir = os.path.join(outdir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    part_name = os.path.join(tar_dir, "*.tar.gz.part_*")
    tmp_tar = os.path.join(tmpdir, "tempfile.tar.gz")
    subprocess.run(f"cat {part_name} > {tmp_tar}", shell=True)

    # Extract tar file
    weightdir = os.path.join(outdir, "weights")
    os.makedirs(weightdir, exist_ok=True)
    subprocess.run(f"tar xzf {tmp_tar} -C {weightdir}", shell=True)

    # Remove temporary files
    os.remove(tmp_tar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["create", "extract"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--maxsize", type=str, default="2G")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "create":
        create_tar_file(args.checkpoint_path, args.outdir, args.maxsize)

    elif args.mode == "extract":
        extract_tar_file(args.checkpoint_path, args.outdir)
