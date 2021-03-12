import argparse
import getpass
import glob
import json
import os
import shutil
from typing import Dict, Iterator, List, NewType, Optional, Tuple
from io import BytesIO
import time

import requests
import torch
import torchvision
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import tqdm

from common import area, load_coco_data, minpt_wh_to_points

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEFAULT_TEMPLATE_LOCATION = "template.json"

CVAT_SERVER_URL = "http://localhost:8080"
API = "/api/v1"
TASKS_PATH = "/tasks"
TIME_TO_SLEEP_AFTER_UPLOAD = 10


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


def upload_to_cvat(
    username: str,
    passwd: str,
    new_img_dir: str,
    image_files: List[str],
    annotation_filepath: str,
    cvat_labels_filepath: str = "settings.json",
) -> None:
    create_task_url = CVAT_SERVER_URL + API + TASKS_PATH

    label_data = json.load(open(cvat_labels_filepath, "r"))
    task_name = new_img_dir.split("/")[-2]
    create_task_body = {"name": task_name, "labels": label_data}

    create_task_response = requests.post(
        create_task_url, json=create_task_body, auth=(username, passwd)
    )
    try:
        assert create_task_response.ok
        print(f"Task {task_name} was created successfully. Uploading images...")
    except AssertionError:
        print(create_task_response.content)
        print(f"Error creating task {task_name}. Exiting.")
        return

    create_task_json = create_task_response.json()
    new_task_url = create_task_json["url"]
    create_task_data_url = new_task_url + "/data"

    create_task_data_body = {"name": task_name, "labels": label_data}
    files = {}
    for i, image_filepath in enumerate(image_files):
        filename = image_filepath.split("/")[-1]
        img_bytes = open(image_filepath, "rb")
        files[f"client_files[{i}]"] = (filename, img_bytes)
    create_task_data_body["image_quality"] = 70
    create_task_data_response = requests.post(
        create_task_data_url,
        data=create_task_data_body,
        auth=(username, passwd),
        files=files,
    )

    try:
        assert create_task_data_response.ok
        print(
            f"{len(image_files)} images uploaded to {task_name}. Adding annotations after {TIME_TO_SLEEP_AFTER_UPLOAD} seconds for server to synchronize..."
        )
    except AssertionError:
        print(create_task_response.content)
        print(f"Error uploading images to task {task_name}. Exiting.")
        return

    time.sleep(TIME_TO_SLEEP_AFTER_UPLOAD)

    upload_annotations_url = new_task_url + "/annotations?format=COCO 1.0"
    coco_annotations_file_content = open(annotation_filepath, "rb")
    upload_annotations_body = {
        "annotation_file": ("instances_default.json", coco_annotations_file_content)
    }
    upload_annotations_response = requests.put(
        upload_annotations_url, files=upload_annotations_body, auth=(username, passwd)
    )
    try:
        assert upload_annotations_response.ok
        print(f"Annotations uploaded to task {task_name}.")
    except AssertionError:
        print(f"Error uploading annotations to task {task_name}. Exiting.")
        return


def bootstrap(
    root_dir: str,
    model_path: str,
    certainty: float,
    template_location: str,
    debug: bool,
) -> None:
    """
    Moves all of the images in `image_dir` into <image_dir>/images, and then uses the PyTorch model at `model_path` to automatically generate annotations for all of the images.
    Annotations will be stored in <image_dir>/annotations.
    """
    new_image_dir = move_images(root_dir)
    image_files = list(glob.iglob(f"{new_image_dir}/*"))
    image_files = sorted(
        [(int(x.split("/")[-1].split(".")[0]), x) for x in image_files],
        key=lambda x: x[0],
    )
    _, image_files = zip(*image_files)
    to_tensor = torchvision.transforms.ToTensor()
    annotation_id = 0
    images = []
    annotations = []
    model: torchvision.models.detection.FasterRCNN = torch.load(model_path)
    for image_file in tqdm.tqdm(image_files, total=len(image_files)):
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
        if debug:
            print("BOXES =", boxes)
            print("LABELS =", labels)
            print("SCORES = ", scores)
            visualize_prediction(img, boxes, labels, scores, certainty)
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)
        for box, label, score in zip(boxes, labels, scores):
            annotation = construct_annotation(
                image_id, annotation_id, box, label, score, certainty
            )
            if annotation is None:
                continue
            annotations.append(annotation)
            annotation_id += 1
        if debug:
            fig.show()
            plt.waitforbuttonpress()
            plt.close(fig)

    licenses, _, info, categories, _ = load_coco_data(template_location)

    new_annotation_dir = os.path.join(root_dir, "annotations")
    if not os.path.exists(new_annotation_dir):
        os.mkdir(new_annotation_dir)
    new_annotation_file_path = os.path.join(
        new_annotation_dir, "instances_default.json"
    )
    with open(new_annotation_file_path, "w+") as annotation_fp:
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

    user_wants_to_upload = input(
        "Would you like to upload the annotated dataset to CVAT? ([Y]/n) "
    )
    if user_wants_to_upload.lower() == "n":
        return
    username = input("What is your username on CVAT? ")
    passwd = getpass.getpass("What is your password on CVAT? ")
    upload_to_cvat(
        username, passwd, new_image_dir, image_files, new_annotation_file_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uses the provided Faster-RCNN model to automatically generate bounding boxes for Smash Bros. characters."
    )
    parser.add_argument(
        "model_path", type=str, help="The path to the F-RCNN model file."
    )
    parser.add_argument(
        "image_dirs", type=str, help="The directory containing images to analyze.", nargs='+'
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
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Whether to visualize predictions and output annotations.",
    )
    args = parser.parse_args()
    for image_dir in args.image_dirs:
        bootstrap(
            image_dir,
            args.model_path,
            args.certainty,
            args.template_location,
            args.debug,
        )
