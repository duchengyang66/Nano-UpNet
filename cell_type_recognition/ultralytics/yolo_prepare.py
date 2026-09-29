import os
import glob
import argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO


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


YOLO_TO_PIX2PIX = {
    0: 1,    # SKBR3
    1: 8,    # PC3
    2: 2,    # 3LL
    3: 3,    # 4T1
    4: 7,    # A375
    5: 12,   # B16
    6: 9,    # MB49
    7: 11,   # SW1990
    8: 5,    # MCF
    9: 16,   # HepG2
    10: 13,  # Hep3B
    11: 18,  # Pan02
    12: 14,  # Hela
    13: 17,  # K180
    14: 4,   # Hct116
    15: 10,  # A549
    16: 20,  # MC38
    17: 19,  # SCLC
    18: 6,   # CT26
    19: 15,  # 231
}


# ============================================================
# YOLO class names
# ============================================================
YOLO_CLASS_NAMES = {
    0: "SKBR3",
    1: "PC3",
    2: "3LL",
    3: "4T1",
    4: "A375",
    5: "B16",
    6: "MB49",
    7: "SW1990",
    8: "MCF",
    9: "HepG2",
    10: "Hep3B",
    11: "Pan02",
    12: "Hela",
    13: "K180",
    14: "Hct116",
    15: "A549",
    16: "MC38",
    17: "SCLC",
    18: "CT26",
    19: "231",
}


# ============================================================
# Find all input images recursively
# ============================================================
def find_images(input_dir):
    image_files = []

    for ext in IMAGE_EXTENSIONS:

        pattern = os.path.join(
            input_dir,
            "**",
            f"*.{ext}"
        )

        image_files.extend(
            glob.glob(
                pattern,
                recursive=True
            )
        )

        pattern_upper = os.path.join(
            input_dir,
            "**",
            f"*.{ext.upper()}"
        )

        image_files.extend(
            glob.glob(
                pattern_upper,
                recursive=True
            )
        )

    return sorted(set(image_files))


# ============================================================
# Determine image-level cell type by majority vote
#
# The cell type with the largest number of detected cells
# is used as the cell type of the whole image.
# ============================================================
def get_majority_cell_type(result):

    if result.boxes is None:
        return None

    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    if len(classes) == 0:
        return None

    unique_classes, counts = np.unique(
        classes,
        return_counts=True
    )

    majority_class = unique_classes[
        np.argmax(counts)
    ]

    return YOLO_CLASS_NAMES[
        int(majority_class)
    ]


# ============================================================
# Convert YOLO result to [0, class, instance]
#
# RGB channels:
#   R = 0
#   G = Pix2PixHD class ID (1~20)
#   B = instance ID (1~255)
#
# Background:
#   [0, 0, 0]
# ============================================================
def create_label_from_yolo(
    result,
    image_size
):

    width, height = image_size

    label = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    if result.masks is None:
        return label, 0

    masks = result.masks.data
    classes = result.boxes.cls

    masks = (
        masks
        .detach()
        .cpu()
        .numpy()
    )

    classes = (
        classes
        .detach()
        .cpu()
        .numpy()
    )

    num_cells = len(masks)

    if num_cells == 0:
        return label, 0

    # ========================================================
    # Process each detected cell
    # ========================================================
    for instance_idx, (
        mask,
        yolo_class
    ) in enumerate(
        zip(masks, classes),
        start=1
    ):

        yolo_class = int(yolo_class)

        if yolo_class not in YOLO_TO_PIX2PIX:
            raise ValueError(
                f"Unknown YOLO class ID: {yolo_class}"
            )

        # Correct YOLO -> Pix2PixHD mapping
        pix2pix_class = YOLO_TO_PIX2PIX[
            yolo_class
        ]

        if (
            pix2pix_class < 1
            or pix2pix_class > 20
        ):
            raise ValueError(
                f"Invalid Pix2PixHD class ID: "
                f"{pix2pix_class}"
            )

        if instance_idx > 255:
            raise ValueError(
                f"Detected {num_cells} cells. "
                f"Instance ID exceeds 255."
            )

        # ----------------------------------------------------
        # Resize mask to original image size
        # ----------------------------------------------------
        mask_img = Image.fromarray(
            (mask * 255).astype(
                np.uint8
            )
        )

        mask_img = mask_img.resize(
            (width, height),
            Image.NEAREST
        )

        mask_binary = (
            np.array(mask_img) > 127
        )

        # ----------------------------------------------------
        # Write [0, class, instance]
        # ----------------------------------------------------
        label[
            mask_binary,
            0
        ] = 0

        label[
            mask_binary,
            1
        ] = pix2pix_class

        label[
            mask_binary,
            2
        ] = instance_idx

    return label, num_cells


# ============================================================
# Save label PNG
# ============================================================
def save_label(
    label,
    output_path
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    Image.fromarray(
        label,
        mode="RGB"
    ).save(
        output_path
    )


# ============================================================
# Process one image
# ============================================================
def process_one_image(
    model,
    image_path,
    input_dir,
    output_dir,
    imgsz,
    conf,
    device
):

    # --------------------------------------------------------
    # Relative path
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Output label path
    # --------------------------------------------------------
    label_filename = (
        stem + "_label.png"
    )

    label_dir = os.path.join(
        output_dir,
        relative_dir
    )

    label_path = os.path.join(
        label_dir,
        label_filename
    )

    # --------------------------------------------------------
    # Read original image size
    # --------------------------------------------------------
    with Image.open(
        image_path
    ) as img:

        image_size = (
            img
            .convert("RGB")
            .size
        )

    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------
    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False
    )

    if len(results) == 0:
        return

    result = results[0]

    # --------------------------------------------------------
    # Get majority cell type
    # --------------------------------------------------------
    majority_cell_type = (
        get_majority_cell_type(result)
    )

    # --------------------------------------------------------
    # Print ONLY the majority cell type
    # --------------------------------------------------------
    if majority_cell_type is not None:
        print(
            majority_cell_type,
            flush=True
        )

    # --------------------------------------------------------
    # Create label
    # --------------------------------------------------------
    label, num_cells = (
        create_label_from_yolo(
            result,
            image_size
        )
    )

    # --------------------------------------------------------
    # Save label
    # --------------------------------------------------------
    save_label(
        label,
        label_path
    )


# ============================================================
# Main
# ============================================================
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Nano-UpNet YOLOv8 preprocessing: "
            "bright-field image -> "
            "[0,class,instance] label PNG"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------
    if not os.path.isdir(
        args.input
    ):
        raise FileNotFoundError(
            f"Input directory does not exist:\n"
            f"{args.input}"
        )

    # --------------------------------------------------------
    # Load YOLO model
    # --------------------------------------------------------
    model = YOLO(
        args.weights,
        task="segment"
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------
    image_files = find_images(
        args.input
    )

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No supported images found in:\n"
            f"{args.input}"
        )

    # --------------------------------------------------------
    # Process all images
    # --------------------------------------------------------
    for image_path in image_files:

        try:

            process_one_image(
                model=model,
                image_path=image_path,
                input_dir=args.input,
                output_dir=args.output,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device
            )

        except Exception as e:

            print(
                f"ERROR: {e}",
                flush=True
            )


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    main()