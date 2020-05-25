import argparse
import functools
import json
import logging
import os
from pprint import pprint
from typing import Dict, List, NamedTuple, Optional, Tuple
from matplotlib import pyplot as plt
import skimage.io as io

from pycocotools.coco import COCO

MIN_NUM_DATASETS = 2


def load_dataset(dataset_path: str):
    """
    Loads a COCO dataset.
    """
    annotation_filepath = os.path.join(
        dataset_path, "annotations", "instances_default.json"
    )
    return COCO(annotation_filepath)


def update_image(
    start_id: int, old_img_dir: str, new_img_dir: str, image: Dict
) -> Dict:
    old_filepath = os.path.join(old_img_dir, "images", f"{image['id']}.jpg")
    image["id"] += start_id
    new_filepath = os.path.join(new_img_dir, "images", f"{image['id']}.jpg")
    os.rename(old_filepath, new_filepath)
    image["file_name"] = f"{image['id']}.jpg"
    return image


def update_annotation(
    start_image_id: int,
    start_annotation_id: int,
    old_img_dir: str,
    new_img_dir: str,
    annotation: Dict,
) -> Dict:
    annotation["id"] += start_annotation_id
    annotation["image_id"] += start_image_id
    return annotation


def assert_categories_are_equal(a_cats: Dict, b_cats: Dict):
    assert len(a_cats) == len(
        b_cats
    ), "The number of categories must be identical in both COCO datasets."

    a_cat_keys = sorted(a_cats.keys())
    b_cat_keys = sorted(b_cats.keys())

    for a_cat_key, b_cat_key in zip(a_cat_keys, b_cat_keys):
        assert (
            a_cat_key == b_cat_key
        ), f"There are different category IDs present in the provided datasets: ({a_cat_key} != {b_cat_key})"
        a_cat_name = a_cats[a_cat_key]["name"]
        b_cat_name = b_cats[b_cat_key]["name"]
        assert (
            a_cat_name == b_cat_name
        ), f"Differing categories detected: ({a_cat_key} -> {a_cat_name} != {b_cat_key} -> {b_cat_name})"


def merge_datasets(
    min_coco_tuple: Tuple[COCO, str], max_coco_tuple: Tuple[COCO, str]
) -> COCO:
    min_coco, min_coco_img_dir = min_coco_tuple
    max_coco, max_coco_img_dir = max_coco_tuple

    assert_categories_are_equal(min_coco.cats, max_coco.cats)

    print(min_coco, max_coco)

    # We're merging min_coco into max_coco, so how many annotations/images are in max_coco?
    num_imgs = len(max_coco.imgs)
    num_anns = len(max_coco.anns)

    # Append images in min_coco to max_coco
    img_ids = min_coco.imgs.keys()
    images = min_coco.loadImgs(ids=img_ids)
    images = map(
        lambda x: update_image(num_imgs, min_coco_img_dir, max_coco_img_dir, x), images
    )
    max_coco.dataset["images"] += images

    # Append annotations in min_coco to max_coco
    ann_ids = min_coco.anns.keys()
    annotations = min_coco.loadAnns(ids=ann_ids)
    annotations = list(
        map(
            lambda x: update_annotation(
                num_imgs, num_anns, min_coco_img_dir, max_coco_img_dir, x
            ),
            annotations,
        )
    )
    max_coco.dataset["annotations"] += annotations
    max_coco.createIndex()

    # Test last image annotations transferred correctly
    last_img_id = list(max_coco.imgs.keys())[-1]
    last_img = max_coco.loadImgs(ids=[last_img_id])[0]
    last_img_filepath = os.path.join(max_coco_img_dir, "images", last_img["file_name"])
    img = io.imread(last_img_filepath)
    plt.figure()
    plt.imshow(img)

    anns = max_coco.loadAnns(max_coco.getAnnIds(imgIds=[last_img_id]))
    max_coco.showAnns(anns, draw_bbox=True)
    plt.show()

    return max_coco


def main(dataset_paths: List[str]):
    datasets = list(map(load_dataset, dataset_paths))

    zipped = sorted(zip(datasets, dataset_paths), key=lambda x: len(x[0].imgs))

    mega_dataset = functools.reduce(merge_datasets, zipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merges COCO datasets together.")
    parser.add_argument(
        "dataset_paths", nargs="*", help="Paths to the datasets to merge together."
    )
    args = parser.parse_args()

    if len(args.dataset_paths) < MIN_NUM_DATASETS:
        raise ValueError(
            f"At least {MIN_NUM_DATASETS} datasets must be provided. Received {len(args.dataset_paths)}."
        )
    main(args.dataset_paths)
