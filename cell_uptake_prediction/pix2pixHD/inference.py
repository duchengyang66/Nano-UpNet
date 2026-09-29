import os
import glob
import argparse

import numpy as np
from PIL import Image

import torch

from options.test_options import TestOptions
from models.models import create_model


# ============================================================
# Supported image extensions
# ============================================================
IMAGE_EXTENSIONS = [
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff",
    "webp",
]


# ============================================================
# Find all bright-field images recursively
# ============================================================
def find_images(input_dir):

    image_files = []

    for ext in IMAGE_EXTENSIONS:

        image_files.extend(
            glob.glob(
                os.path.join(
                    input_dir,
                    "**",
                    f"*.{ext}"
                ),
                recursive=True
            )
        )

        image_files.extend(
            glob.glob(
                os.path.join(
                    input_dir,
                    "**",
                    f"*.{ext.upper()}"
                ),
                recursive=True
            )
        )

    return sorted(set(image_files))


# ============================================================
# Find corresponding label
#
# Example:
#
# BF:
#   abc.png
#
# Label:
#   abc_label.png
# ============================================================
def find_label_path(
    image_path,
    input_dir,
    label_dir
):

    relative_path = os.path.relpath(
        image_path,
        input_dir
    )

    relative_dir = os.path.dirname(
        relative_path
    )

    filename = os.path.basename(
        relative_path
    )

    stem = os.path.splitext(
        filename
    )[0]

    label_filename = stem + "_label.png"

    label_path = os.path.join(
        label_dir,
        relative_dir,
        label_filename
    )

    return label_path


# ============================================================
# Convert image to tensor
#
# Output:
# [1, 3, H, W]
#
# Same normalization convention commonly used by Pix2PixHD:
# uint8 [0,255] -> float [-1,1]
# ============================================================
def image_to_tensor(image):

    image = image.convert("RGB")

    array = np.asarray(
        image,
        dtype=np.float32
    )

    tensor = torch.from_numpy(
        array
    ).permute(
        2, 0, 1
    )

    tensor = tensor / 127.5 - 1.0

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# Label to tensor
#
# IMPORTANT:
# Label itself contains integer IDs:
#
# R = 0
# G = class
# B = instance
#
# We first convert to float and then apply the SAME transform
# convention used by the existing dataset.
# ============================================================
def label_to_tensor(label):

    label = label.convert("RGB")

    array = np.asarray(
        label,
        dtype=np.float32
    )

    tensor = torch.from_numpy(
        array
    ).permute(
        2, 0, 1
    )

    tensor = tensor / 127.5 - 1.0

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# Save generated image
# ============================================================
def save_generated_image(
    tensor,
    output_path
):

    # tensor:
    # [1, 3, H, W]
    image = tensor.detach().cpu()[0]

    # [-1,1] -> [0,255]
    image = (
        (image + 1.0)
        * 127.5
    )

    image = torch.clamp(
        image,
        0,
        255
    )

    image = image.permute(
        1, 2, 0
    ).numpy().astype(
        np.uint8
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    Image.fromarray(
        image,
        mode="RGB"
    ).save(
        output_path
    )


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Nano-UpNet Pix2PixHD inference"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Bright-field image directory"
    )

    parser.add_argument(
        "--labels",
        required=True,
        type=str,
        help="YOLO label directory"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="w_flip_allcells_20_8",
        help="Pix2PixHD experiment/checkpoint name"
    )

    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="Pix2PixHD checkpoints directory"
    )

    parser.add_argument(
        "--which_epoch",
        type=str,
        default="latest",
        help="Checkpoint epoch"
    )

    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="GPU ID"
    )

    args = parser.parse_args()

    # ========================================================
    # Make command-line arguments available to TestOptions
    # ========================================================
    import sys

    original_argv = sys.argv

    sys.argv = [
        "pix2pix_inference.py",

        "--name",
        args.name,

        "--checkpoints_dir",
        args.checkpoints_dir,

        "--which_epoch",
        args.which_epoch,

        "--gpu_ids",
        args.gpu_ids,

        "--label_nc",
        "0",

        "--input_nc",
        "6",

        "--output_nc",
        "3",

        "--no_instance",

        "--resize_or_crop",
        "none",

        "--loadSize",
        "1024",

        "--fineSize",
        "1024",

        "--netG",
        "local",

        "--n_local_enhancers",
        "1",

        "--batchSize",
        "1",

        "--how_many",
        "999999",

    ]

    opt = TestOptions().parse(
        save=False
    )

    sys.argv = original_argv

    # ========================================================
    # Force inference settings
    # ========================================================
    opt.nThreads = 1
    opt.batchSize = 1
    opt.serial_batches = True
    opt.no_flip = True

    # ========================================================
    # Load Pix2PixHD model
    # ========================================================
    print("=" * 70)
    print("Loading Pix2PixHD model")
    print("=" * 70)

    print(
        f"Name        : {opt.name}"
    )

    print(
        f"Epoch       : {opt.which_epoch}"
    )

    print(
        f"Checkpoints : {opt.checkpoints_dir}"
    )

    print(
        f"Input       : {args.input}"
    )

    print(
        f"Labels      : {args.labels}"
    )

    print(
        f"Output      : {args.output}"
    )

    print("=" * 70)

    model = create_model(
        opt
    )

    model.eval()

    # ========================================================
    # Find BF images
    # ========================================================
    image_files = find_images(
        args.input
    )

    if len(image_files) == 0:

        raise FileNotFoundError(
            "No supported images found in:\n"
            + args.input
        )

    print(
        f"Found {len(image_files)} "
        f"bright-field image(s)."
    )

    # ========================================================
    # Process images
    # ========================================================
    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"\n[{index}/{len(image_files)}]"
        )

        try:

            # ------------------------------------------------
            # Corresponding label
            # ------------------------------------------------
            label_path = find_label_path(
                image_path,
                args.input,
                args.labels
            )

            if not os.path.isfile(
                label_path
            ):

                print(
                    "[WARNING] Label not found:"
                )

                print(
                    label_path
                )

                continue

            # ------------------------------------------------
            # Load BF
            # ------------------------------------------------
            bf_image = Image.open(
                image_path
            ).convert("RGB")

            # ------------------------------------------------
            # Load label
            # ------------------------------------------------
            label_image = Image.open(
                label_path
            ).convert("RGB")

            # ------------------------------------------------
            # Make sure sizes are identical
            # ------------------------------------------------
            if label_image.size != bf_image.size:

                print(
                    "Resizing label from "
                    f"{label_image.size} "
                    f"to "
                    f"{bf_image.size}"
                )

                label_image = label_image.resize(
                    bf_image.size,
                    Image.NEAREST
                )

            # ------------------------------------------------
            # Convert to tensors
            # ------------------------------------------------
            bf_tensor = image_to_tensor(
                bf_image
            )

            label_tensor = label_to_tensor(
                label_image
            )

            # ------------------------------------------------
            # Concatenate:
            #
            # BF:
            # [R,G,B]
            #
            # label:
            # [0,class,instance]
            #
            # Result:
            # [BF_R,
            #  BF_G,
            #  BF_B,
            #  0,
            #  class,
            #  instance]
            # ------------------------------------------------
            model_input = torch.cat(
                [
                    bf_tensor,
                    label_tensor
                ],
                dim=1
            )

            # ------------------------------------------------
            # Move to GPU
            # ------------------------------------------------
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            model_input = model_input.to(
                device
            )

            # ------------------------------------------------
            # Pix2PixHD inference
            #
            # no_instance=True
            # therefore inst=None
            # ------------------------------------------------
            with torch.no_grad():

                generated = model.inference(
                    model_input,
                    None,
                    None
                )

            # ------------------------------------------------
            # Output path
            # ------------------------------------------------
            relative_path = os.path.relpath(
                image_path,
                args.input
            )

            relative_dir = os.path.dirname(
                relative_path
            )

            filename = os.path.basename(
                relative_path
            )

            stem = os.path.splitext(
                filename
            )[0]

            output_filename = (
                stem
                + "_uptake.png"
            )

            output_path = os.path.join(
                args.output,
                relative_dir,
                output_filename
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------
            save_generated_image(
                generated,
                output_path
            )

            print(
                f"Input : {image_path}"
            )

            print(
                f"Label : {label_path}"
            )

            print(
                f"Output: {output_path}"
            )

        except Exception as e:

            print(
                "[ERROR]"
            )

            print(
                f"Image: {image_path}"
            )

            print(
                f"Reason: {e}"
            )

    # ========================================================
    # Finished
    # ========================================================
    print("\n" + "=" * 70)

    print(
        "Pix2PixHD inference finished."
    )

    print(
        f"Results saved to:\n{args.output}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()