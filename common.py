"""
Common functionality for bootstrapping.
"""
import json
import os
from typing import NewType, Tuple
from matplotlib import patches

from pycocotools.coco import COCO

Point = NewType("Point", Tuple[float, float])


def minpt_wh_to_points(
    minpt_wh: Tuple[float, float, float, float]
) -> Tuple[Point, Point, Point, Point]:
    """
    Converts (top-left x, top-left y, width, height) into 4-corner bounding box format: (TL, TR, BR, BL)
    """
    min_x, min_y, width, height = minpt_wh
    top_left = (min_x, min_y)
    top_right = (min_x + width, min_y)
    bottom_right = (min_x + width, min_y + height)
    bottom_left = (min_x, min_y + height)

    return top_left, top_right, bottom_right, bottom_left


def points_to_minpt_wh(
    points: Tuple[Point, Point, Point, Point]
) -> Tuple[float, float, float, float]:
    """
    Converts (TL, TR, BR, BL) point format into (TL.x, TL.y, width, height)
    """
    top_left, _, bottom_right, _ = points
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    return (top_left[0], top_left[1], width, height)


def area(minpt_wh: Tuple[float, float, float, float]):
    """
    Calculates area from points given in (TL.x, TL.y, width, height) format.
    """
    return minpt_wh[2] * minpt_wh[3]


def load_coco_data(annotation_path: str):
    with open(annotation_path, "r") as json_file:
        coco_data = json.load(json_file)
    licenses = coco_data["licenses"]
    images = coco_data["images"]
    info = coco_data["info"]
    categories = coco_data["categories"]
    annotations = coco_data["annotations"]
    return licenses, images, info, categories, annotations

def load_dataset(dataset_path: str) -> COCO:
    """
    Loads a COCO dataset from a specific folder. Expects folder to be similar to:
    <dir>/annotations/instances_default.json
    <dir>/images/
    """
    annotation_filepath = os.path.join(
        dataset_path, "annotations", "instances_default.json"
    )
    return COCO(annotation_filepath)

def add_annotation(ax, annotation) -> None:
    [min_x, min_y, width, height] = annotation["bbox"]
    category_id = annotation["category_id"]

    rect = patches.Rectangle(
        (min_x, min_y), width, height, linewidth=1, edgecolor="g", facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(min_x, min_y, f"{category_id}", c="g")