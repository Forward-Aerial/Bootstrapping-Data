import argparse
import datetime
import glob
import json
import logging
import os
import shutil
from typing import Dict, List, Tuple

import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image
from torchvision import transforms


def construct_annotation(
    image_id: int,
    annotation_id: int,
    box: Tuple[float, float, float, float],
    label: int,
    categories: List[Dict],
) -> Dict:
    min_x, min_y, max_x, max_y = box
    width = max_x - min_x
    height = max_y - min_y
    points = [
        min_x,  # Top Left
        min_y,
        min_x + width,  # Top Right
        min_y,
        min_x + width,  # Bottom Right
        min_y + height,
        min_x,
        min_y + height,  # Bottom Left
    ]

    return {
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": [min_x, min_y, width, height],
        "category_id": label,
        "segmentation": [points],
        "id": annotation_id,
        "area": width * height,
    }


def load_coco_data(annotation_path: str):
    with open(annotation_path, "r") as json_file:
        coco_data = json.load(json_file)
    licenses = coco_data["licenses"]
    images = coco_data["images"]
    info = coco_data["info"]
    categories = coco_data["categories"]
    annotations = coco_data["annotations"]
    return licenses, images, info, categories, annotations


def display_image(img: Image, prediction):
    # Visualize bounding boxes
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    for box, label, score in zip(
        prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"].round()
    ):
        if not score.item():
            continue
        [min_x, min_y, max_x, max_y] = box.tolist()
        width = max_x - min_x
        height = max_y - min_y
        rect = patches.Rectangle(
            (min_x, min_y), width, height, linewidth=1, edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(min_x, min_y, "Captain Falcon" if label.item() == 2 else "Fox", c="r")
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)


def bootstrap(
    existing_image_dir: str,
    existing_annotation_path: str,
    new_images_dir: str,
    model_path: str,
):
    licenses, images, info, categories, annotations = load_coco_data(
        existing_annotation_path
    )

    new_image_paths = glob.iglob(os.path.join(new_images_dir, "*"))
    model: torchvision.models.detection.FasterRCNN = torch.load(model_path)

    to_tensor = transforms.ToTensor()

    max_image_id = max(images, key=lambda x: x["id"])["id"]

    max_annotation_id = max(annotations, key=lambda x: x["id"])["id"]
    annotation_id = max_annotation_id + 1
    for i, image_path in enumerate(new_image_paths):
        image_id = max_image_id + i
        logging.info(f"Processing {image_path}")

        # Copy it to the existing images directory using the ID as a filename
        new_image_location = os.path.join(existing_image_dir, f"{image_id}.jpg")
        logging.debug(f"Copying from {image_path} to {new_image_location}")
        shutil.copyfile(image_path, new_image_location)

        # Update the variable so the rest proceeds as expected.
        image_path = new_image_location

        # Add the image to the list of images
        img: Image = Image.open(image_path)
        width, height = img.size
        images.append(
            {
                "date_captured": 0,
                "flickr_url": 0,
                "height": height,
                "width": width,
                "id": image_id,
                "license": 0,
                "file_name": f"{image_id}.jpg",
            }
        )
        img = Image.open(image_path)
        img_tensor = to_tensor(img).to("cuda")
        prediction = model([img_tensor])
        # display_image(img, prediction)
        boxes = prediction[0]["boxes"]
        labels = prediction[0]["labels"]
        scores = prediction[0]["scores"]
        for box, label, score in zip(boxes, labels, scores):
            if score.item() <= 0.7:
                continue
            annotation = construct_annotation(
                image_id, annotation_id, box.tolist(), label.item(), categories,
            )
            annotations.append(annotation)
            annotation_id += 1

    with open(existing_annotation_path, "w") as annotation_file:
        json.dump(
            {
                "licenses": licenses,
                "images": images,
                "annotations": annotations,
                "info": info,
                "categories": categories,
            },
            annotation_file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "existing_images",
        type=str,
        help="Path to the directory containing existing COCO images",
    )
    parser.add_argument(
        "existing_annotations",
        type=str,
        help="Path to the file containing existing COCO annotations",
    )
    parser.add_argument(
        "new_images",
        type=str,
        help="Path to the directory containing new images to label",
    )
    parser.add_argument(
        "model_file", type=str, help="Path to a trained PyTorch FasterRCNN model file."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    bootstrap(
        args.existing_images,
        args.existing_annotations,
        args.new_images,
        args.model_file,
    )
