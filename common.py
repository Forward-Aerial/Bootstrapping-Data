"""
Common functionality for bootstrapping.
"""
import json
from typing import NewType, Tuple

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
