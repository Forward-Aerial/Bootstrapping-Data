import argparse
import glob
import json
import os
import shutil
from typing import Dict, List, NewType, Optional, Tuple

import torch
import torchvision
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image

from common import area, load_coco_data, minpt_wh_to_points

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEFAULT_TEMPLATE_LOCATION = "template.json"


def move_images(image_dir: str) -> str:
    """
    Moves all of the images located in `image_dir` into <image_dir>/images
    """
    new_image_dir = f"{image_dir}/images"
    if os.path.exists(new_image_dir) and len(os.listdir(new_image_dir)):
        raise Exception(
            f"{new_image_dir} exists and is not empty. Please move the files out of {new_image_dir} and retry."
        )
    image_files = glob.iglob(f"{image_dir}/*")

    for image_id, image_file in enumerate(image_files):
        extension = os.path.basename(image_file).split(".")[1]
        new_filename = f"{image_id}.{extension}"
        os.renames(image_file, os.path.join(new_image_dir, new_filename))
    return new_image_dir


def construct_annotation(
    image_id: int,
    annotation_id: int,
    box: torch.Tensor,
    label: torch.Tensor,
    score: torch.Tensor,
    certainty: float,
) -> Optional[Dict]:
    if score.item() < certainty:
        return None
    [min_x, min_y, max_x, max_y] = box.tolist()
    width = max_x - min_x
    height = max_y - min_y
    minpt_wh = (min_x, min_y, width, height)
    tl, tr, br, bl = minpt_wh_to_points(minpt_wh)
    points = [*tl, *tr, *br, *bl]

    return {
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": [*minpt_wh],
        "category_id": label.item(),
        "segmentation": [points],
        "id": annotation_id,
        "area": area(minpt_wh),
    }


def visualize_prediction(
    img: Image,
    boxes: List[float],
    labels: List[float],
    scores: List[float],
    certainty: float,
) -> None:
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)
    for box, label, score in zip(boxes, labels, scores):
        if score < certainty:
            continue
        [min_x, min_y, max_x, max_y] = box
        width = max_x - min_x
        height = max_y - min_y
        rect = patches.Rectangle(
            (min_x, min_y), width, height, linewidth=1, edgecolor="r", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(min_x, min_y, f"{label} ({score})", c="r")
    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)


def add_annotation(fig, ax, annotation) -> None:

    [min_x, min_y, width, height] = annotation["bbox"]
    category_id = annotation["category_id"]

    rect = patches.Rectangle(
        (min_x, min_y), width, height, linewidth=1, edgecolor="g", facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(min_x, min_y, f"{category_id}", c="g")
    fig.show()


def bootstrap(
    root_dir: str, model_path: str, certainty: float, template_location: str,
) -> None:
    """
    Moves all of the images in `image_dir` into <image_dir>/images, and then uses the PyTorch model at `model_path` to automatically generate annotations for all of the images.
    Annotations will be stored in <image_dir>/annotations.
    """
    new_image_dir = move_images(root_dir)
    image_files = glob.iglob(f"{new_image_dir}/*")
    to_tensor = torchvision.transforms.ToTensor()
    annotation_id = 0
    images = []
    annotations = []
    model: torchvision.models.detection.FasterRCNN = torch.load(model_path)
    for image_file in image_files:
        image_id = int(os.path.basename(image_file).split(".")[0])
        print(image_file, image_id)
        img = Image.open(image_file)
        img_tensor = to_tensor(img).to(DEVICE)
        prediction = model([img_tensor])[0]
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]
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
        # print("BOXES =", boxes)
        # print("LABELS =", labels)
        # print("SCORES = ", scores)
        # visualize_prediction(img, boxes, labels, scores, certainty)
        # fig, ax = plt.subplots(1, figsize=(12, 8))
        # ax.imshow(img)
        for box, label, score in zip(boxes, labels, scores):
            annotation = construct_annotation(
                image_id, annotation_id, box, label, score, certainty
            )
            if annotation is None:
                continue
            # add_annotation(fig, ax, annotation)
            annotations.append(annotation)
            annotation_id += 1
        # fig.show()
        # plt.waitforbuttonpress()
        # plt.close(fig)

    licenses, _, info, categories, _ = load_coco_data(template_location)

    new_annotation_dir = os.path.join(root_dir, "annotations")
    if not os.path.exists(new_annotation_dir):
        os.mkdir(new_annotation_dir)
    with open(
        os.path.join(new_annotation_dir, "instances_default.json"), "w+"
    ) as annotation_fp:
        json.dump(
            {
                "licenses": licenses,
                "info": info,
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            annotation_fp,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uses the provided Faster-RCNN model to automatically generate bounding boxes for Smash Bros. characters."
    )
    parser.add_argument(
        "image_dir", type=str, help="The directory containing images to analyze."
    )
    parser.add_argument(
        "model_path", type=str, help="The path to the F-RCNN model file."
    )
    parser.add_argument(
        "--certainty",
        type=float,
        default=0.25,
        help="Discard bounding boxes under this certainty.",
    )
    parser.add_argument(
        "--template-location",
        type=str,
        default=DEFAULT_TEMPLATE_LOCATION,
        help="The location of the Smash COCO template.",
    )
    args = parser.parse_args()
    bootstrap(args.image_dir, args.model_path, args.certainty, args.template_location)
